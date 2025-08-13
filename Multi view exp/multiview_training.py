#!/usr/bin/env python3
"""
Multi-View YOLO11m Classification Training

This script trains a YOLO11m-cls model on multi-view 2D images generated from 3D volumes,
incorporating view information (azimuth and elevation angles) as additional input features
for improved classification performance.

Features:
- Custom YOLO11m-cls model with view-aware architecture
- Multi-view dataset loader with view angle encoding
- View-conditioned training pipeline
- Comprehensive evaluation and visualization

Usage:
    python multiview_training.py --data_root /path/to/multiview/dataset --epochs 100
    python multiview_training.py --data_root /path/to/multiview/dataset --resume weights/best.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
import cv2
import json
import math
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
from tqdm import tqdm
import yaml

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.nn.modules import Classify
from ultralytics.utils import LOGGER
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViewEncoder:
    """Encode view angles (azimuth, elevation) into feature vectors"""
    
    def __init__(self, encoding_dim: int = 16):
        """
        Args:
            encoding_dim: Dimension of the view encoding vector
        """
        self.encoding_dim = encoding_dim
    
    def encode_view_angles(self, azimuth: float, elevation: float) -> torch.Tensor:
        """
        Encode azimuth and elevation angles into a feature vector
        
        Args:
            azimuth: Azimuth angle in degrees (0-360)
            elevation: Elevation angle in degrees (0-90)
        
        Returns:
            Encoded view vector of shape (encoding_dim,)
        """
        # Convert to radians
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        
        # Create sinusoidal encoding for azimuth (periodic)
        az_encodings = []
        for i in range(self.encoding_dim // 4):
            freq = 2 ** i
            az_encodings.extend([
                math.sin(freq * az_rad),
                math.cos(freq * az_rad)
            ])
        
        # Create sinusoidal encoding for elevation (non-periodic, 0-90)
        el_encodings = []
        for i in range(self.encoding_dim // 4):
            freq = (i + 1) * math.pi / 90  # Scale to elevation range
            el_encodings.extend([
                math.sin(freq * elevation),
                math.cos(freq * elevation)
            ])
        
        # Combine encodings
        encoding = az_encodings + el_encodings
        
        # Pad or truncate to exact dimension
        if len(encoding) > self.encoding_dim:
            encoding = encoding[:self.encoding_dim]
        elif len(encoding) < self.encoding_dim:
            encoding.extend([0.0] * (self.encoding_dim - len(encoding)))
        
        return torch.tensor(encoding, dtype=torch.float32)
    
    def parse_filename_angles(self, filename: str) -> Tuple[float, float]:
        """
        Parse azimuth and elevation from filename
        Expected format: *_az{azimuth:03d}_el{elevation:02d}.png
        
        Args:
            filename: Image filename
            
        Returns:
            Tuple of (azimuth, elevation) angles
        """
        try:
            # Extract angles from filename using the format from 3d_to_multiview.py
            parts = Path(filename).stem.split('_')
            azimuth = None
            elevation = None
            
            for part in parts:
                if part.startswith('az'):
                    azimuth = float(part[2:])
                elif part.startswith('el'):
                    elevation = float(part[2:])
            
            if azimuth is None or elevation is None:
                raise ValueError(f"Could not parse angles from filename: {filename}")
            
            return azimuth, elevation
            
        except Exception as e:
            logger.warning(f"Failed to parse angles from {filename}: {e}")
            # Return default angles if parsing fails
            return 0.0, 45.0


class MultiViewDataset(Dataset):
    """Dataset class for multi-view images with view angle encoding"""
    
    def __init__(self, data_root: str, split: str = 'train', 
                 view_encoding_dim: int = 16, transform=None):
        """
        Args:
            data_root: Root directory containing train/val subdirectories
            split: 'train' or 'val'
            view_encoding_dim: Dimension of view encoding vector
            transform: Image transformations
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.view_encoder = ViewEncoder(view_encoding_dim)
        
        # Load dataset
        self.samples = self._load_samples()
        self.classes = self._get_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all image samples with their labels and view angles"""
        samples = []
        split_path = self.data_root / self.split
        
        if not split_path.exists():
            raise ValueError(f"Split directory not found: {split_path}")
        
        # Each class directory contains multi-view images
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Find all PNG images in the class directory
            for img_path in class_dir.glob('*.png'):
                try:
                    # Parse view angles from filename
                    azimuth, elevation = self.view_encoder.parse_filename_angles(img_path.name)
                    
                    samples.append({
                        'image_path': img_path,
                        'class_name': class_name,
                        'azimuth': azimuth,
                        'elevation': elevation
                    })
                    
                except Exception as e:
                    logger.warning(f"Skipping {img_path}: {e}")
                    continue
        
        return samples
    
    def _get_classes(self) -> List[str]:
        """Get sorted list of class names"""
        classes = set()
        for sample in self.samples:
            classes.add(sample['class_name'])
        return sorted(list(classes))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            Tuple of (image, view_encoding, class_label)
        """
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            raise ValueError(f"Could not load image: {sample['image_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: resize and normalize
            image = cv2.resize(image, (640, 640))
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Encode view angles
        view_encoding = self.view_encoder.encode_view_angles(
            sample['azimuth'], sample['elevation']
        )
        
        # Get class label
        class_idx = self.class_to_idx[sample['class_name']]
        
        return image, view_encoding, class_idx


class ViewAwareYOLO(nn.Module):
    """YOLO11m-cls model with view-aware architecture"""
    
    def __init__(self, num_classes: int, view_encoding_dim: int = 16, 
                 yolo_model_path: str = 'yolo11m-cls.pt'):
        """
        Args:
            num_classes: Number of classification classes
            view_encoding_dim: Dimension of view encoding vector
            yolo_model_path: Path to pre-trained YOLO model
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.view_encoding_dim = view_encoding_dim
        
        # Load pre-trained YOLO11m-cls model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Extract the backbone and neck from YOLO
        self.backbone = self.yolo_model.model.model[:8]  # Backbone layers
        self.neck = self.yolo_model.model.model[8]       # Feature pyramid neck
        
        # Get feature dimension from the last backbone layer
        # For YOLO11m, this is typically 768 or 1024
        self.feature_dim = self._get_backbone_output_dim()
        
        # View processing layers
        self.view_processor = nn.Sequential(
            nn.Linear(view_encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256)
        )
        
        # Fusion layer to combine image features and view features
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_custom_layers()
    
    def _get_backbone_output_dim(self) -> int:
        """Get the output dimension of the backbone"""
        # Create dummy input to get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            features = self.backbone(dummy_input)
            
            # Pool the spatial dimensions
            if len(features.shape) == 4:  # [B, C, H, W]
                pooled = F.adaptive_avg_pool2d(features, (1, 1))
                return pooled.shape[1]
            else:
                return features.shape[-1]
    
    def _init_custom_layers(self):
        """Initialize weights for custom layers"""
        for module in [self.view_processor, self.fusion_layer, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, images: torch.Tensor, view_encodings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            images: Batch of images [B, 3, H, W]
            view_encodings: Batch of view encodings [B, view_encoding_dim]
            
        Returns:
            Classification logits [B, num_classes]
        """
        batch_size = images.size(0)
        
        # Extract image features using YOLO backbone
        image_features = self.backbone(images)
        
        # Global average pooling to get fixed-size features
        if len(image_features.shape) == 4:  # [B, C, H, W]
            image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
            image_features = image_features.view(batch_size, -1)
        
        # Process view encodings
        view_features = self.view_processor(view_encodings)
        
        # Fuse image and view features
        combined_features = torch.cat([image_features, view_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits


class MultiViewTrainer:
    """Training pipeline for view-aware multi-view classification"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ViewAwareYOLO(
            num_classes=config['num_classes'],
            view_encoding_dim=config['view_encoding_dim'],
            yolo_model_path=config.get('yolo_model_path', 'yolo11m-cls.pt')
        ).to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup training components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        
        # Setup logging
        if config.get('use_wandb', False):
            wandb.init(project="multiview-classification", config=config)
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders"""
        # Data transforms
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = MultiViewDataset(
            self.config['data_root'], 
            split='train',
            view_encoding_dim=self.config['view_encoding_dim'],
            transform=train_transform
        )
        
        val_dataset = MultiViewDataset(
            self.config['data_root'],
            split='val', 
            view_encoding_dim=self.config['view_encoding_dim'],
            transform=val_transform
        )
        
        # Update config with dataset info
        self.config['num_classes'] = len(train_dataset.classes)
        self.config['classes'] = train_dataset.classes
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, view_encodings, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            view_encodings = view_encodings.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, view_encodings)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, view_encodings, labels in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                images = images.to(self.device)
                view_encodings = view_encodings.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, view_encodings)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best_model.pt', val_metrics)
                logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch
                })
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Generate final evaluation report
        self.generate_evaluation_report()
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics': metrics
        }
        
        os.makedirs('weights', exist_ok=True)
        torch.save(checkpoint, f'weights/{filename}')
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report with visualizations"""
        logger.info("Generating evaluation report...")
        
        # Load best model
        try:
            self.load_checkpoint('weights/best_model.pt')
        except:
            logger.warning("Could not load best model, using current model")
        
        # Get validation predictions
        val_metrics = self.validate()
        predictions = val_metrics['predictions']
        labels = val_metrics['labels']
        
        # Create evaluation directory
        eval_dir = Path('evaluation')
        eval_dir.mkdir(exist_ok=True)
        
        # Classification report
        class_names = self.config['classes']
        report = classification_report(
            labels, predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # Save classification report
        with open(eval_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(eval_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, per_class_acc * 100)
        plt.title('Per-Class Accuracy')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Class')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{acc*100:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(eval_dir / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # View angle analysis
        self.analyze_view_performance()
        
        logger.info(f"Evaluation report saved to {eval_dir}")
        logger.info(f"Overall accuracy: {val_metrics['accuracy']:.2f}%")
        
        # Print per-class metrics
        print("\nPer-class Performance:")
        print("-" * 50)
        for i, class_name in enumerate(class_names):
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            print(f"{class_name:15} | P: {precision:.3f} | R: {recall:.3f} | "
                  f"F1: {f1:.3f} | Samples: {support:3d}")
    
    def analyze_view_performance(self):
        """Analyze performance across different view angles"""
        logger.info("Analyzing view angle performance...")
        
        self.model.eval()
        view_results = {}
        
        # Collect predictions for each view angle
        with torch.no_grad():
            for batch_idx, (images, view_encodings, labels) in enumerate(self.val_loader):
                images = images.to(self.device)
                view_encodings = view_encodings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, view_encodings)
                _, predictions = outputs.max(1)
                
                # Group by view angles (approximate)
                for i in range(len(labels)):
                    view_enc = view_encodings[i].cpu().numpy()
                    
                    # Decode approximate angles from encoding (simplified)
                    # This is a rough approximation - in practice, you might want to
                    # store the original angles separately for analysis
                    azimuth_approx = int((view_enc[0] * 180 + 180) / 45) * 45  # Rough approximation
                    elevation_approx = int((view_enc[4] * 45 + 45) / 15) * 15   # Rough approximation
                    
                    view_key = f"az_{azimuth_approx:03d}_el_{elevation_approx:02d}"
                    
                    if view_key not in view_results:
                        view_results[view_key] = {'correct': 0, 'total': 0}
                    
                    view_results[view_key]['total'] += 1
                    if predictions[i] == labels[i]:
                        view_results[view_key]['correct'] += 1
        
        # Calculate accuracy per view
        view_accuracies = {}
        for view_key, results in view_results.items():
            if results['total'] > 0:
                accuracy = results['correct'] / results['total'] * 100
                view_accuracies[view_key] = accuracy
        
        # Sort by accuracy
        sorted_views = sorted(view_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        # Save view analysis
        eval_dir = Path('evaluation')
        with open(eval_dir / 'view_analysis.json', 'w') as f:
            json.dump(view_accuracies, f, indent=2)
        
        # Plot view performance
        if len(sorted_views) > 0:
            views, accuracies = zip(*sorted_views[:20])  # Top 20 views
            
            plt.figure(figsize=(15, 8))
            bars = plt.bar(range(len(views)), accuracies)
            plt.title('Accuracy by View Angle (Top 20)')
            plt.ylabel('Accuracy (%)')
            plt.xlabel('View Angle')
            plt.xticks(range(len(views)), views, rotation=45, ha='right')
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(eval_dir / 'view_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Best performing view: {sorted_views[0][0]} ({sorted_views[0][1]:.1f}%)")
            logger.info(f"Worst performing view: {sorted_views[-1][0]} ({sorted_views[-1][1]:.1f}%)")


def visualize_view_encodings():
    """Utility function to visualize how view angles are encoded"""
    encoder = ViewEncoder(encoding_dim=16)
    
    # Generate encodings for different view angles
    angles = []
    encodings = []
    
    for azimuth in range(0, 360, 45):
        for elevation in range(15, 91, 30):
            angles.append((azimuth, elevation))
            encoding = encoder.encode_view_angles(azimuth, elevation)
            encodings.append(encoding.numpy())
    
    encodings = np.array(encodings)
    
    # Plot encoding visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap of encodings
    im1 = ax1.imshow(encodings, cmap='viridis', aspect='auto')
    ax1.set_title('View Angle Encodings')
    ax1.set_xlabel('Encoding Dimension')
    ax1.set_ylabel('View Configuration')
    ax1.set_yticks(range(0, len(angles), 2))
    ax1.set_yticklabels([f"Az{a[0]}°El{a[1]}°" for a in angles[::2]], fontsize=8)
    plt.colorbar(im1, ax=ax1)
    
    # PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    encoded_2d = pca.fit_transform(encodings)
    
    # Color by elevation
    elevations = [a[1] for a in angles]
    scatter = ax2.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=elevations, 
                         cmap='coolwarm', s=60)
    ax2.set_title('View Encodings (PCA projection)')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add angle labels
    for i, (az, el) in enumerate(angles):
        ax2.annotate(f'{az}°/{el}°', (encoded_2d[i, 0], encoded_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=6)
    
    plt.colorbar(scatter, ax=ax2, label='Elevation (degrees)')
    plt.tight_layout()
    plt.savefig('view_encoding_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_sample_config():
    """Create a sample YAML configuration file"""
    config = create_default_config()
    config['data_root'] = './multiview_dataset'
    config['epochs'] = 100
    config['batch_size'] = 16
    config['learning_rate'] = 1e-4
    config['use_wandb'] = False
    
    with open('multiview_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Sample configuration saved to multiview_config.yaml")


def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        'data_root': './multiview_dataset',
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'view_encoding_dim': 16,
        'yolo_model_path': 'yolo11m-cls.pt',
        'use_wandb': False,
        'num_classes': None,  # Will be set automatically
        'classes': None       # Will be set automatically
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-View YOLO11m Classification Training')
    
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing train/val subdirectories')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    config['data_root'] = args.data_root
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['use_wandb'] = args.wandb
    
    # Validate data directory
    if not os.path.exists(config['data_root']):
        logger.error(f"Data directory not found: {config['data_root']}")
        return 1
    
    try:
        # Initialize trainer
        trainer = MultiViewTrainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Start training
        trainer.train()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())