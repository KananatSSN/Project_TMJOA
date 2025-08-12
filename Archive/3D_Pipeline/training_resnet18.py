import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
import torch.nn.functional as F


class NiftiDataset(Dataset):
    """Custom dataset for loading .nii.gz files"""
    
    def __init__(self, data_dir, transform=None, target_size=(64, 64, 64)):
        """
        Args:
            data_dir: Directory containing class folders (0/ and 1/)
            transform: Optional transform to be applied on a sample
            target_size: Target size to resize images to (depth, height, width)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = []
        
        # Load file paths and labels
        for class_label in ['0', '1']:
            class_dir = os.path.join(data_dir, class_label)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.nii.gz') or filename.endswith('.nii'):
                        filepath = os.path.join(class_dir, filename)
                        self.samples.append((filepath, int(class_label)))
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        # Load NIfTI image
        nii_img = nib.load(filepath)
        image = nii_img.get_fdata()
        
        # Handle different input dimensions
        if len(image.shape) == 4:
            # If 4D, take the first volume
            image = image[:, :, :, 0]
        
        # Normalize to [0, 1]
        image = self._normalize_image(image)
        
        # Resize if needed
        if image.shape != self.target_size:
            image = self._resize_image(image, self.target_size)
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)  # Shape: (1, D, H, W)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _normalize_image(self, image):
        """Normalize image to [0, 1] range"""
        # Remove NaN and infinity values
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to [0, 1]
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
        
        return image
    
    def _resize_image(self, image, target_size):
        """Resize image using trilinear interpolation"""
        # Convert to tensor for interpolation
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            image_tensor, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        )
        
        return resized.squeeze(0).squeeze(0).numpy()


class SimpleTransform:
    """Simple data augmentation for 3D images"""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        # Random horizontal flip
        if torch.rand(1) < self.prob:
            image = torch.flip(image, dims=[3])  # Flip width
        
        # Random rotation (90 degrees)
        if torch.rand(1) < self.prob:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k=k, dims=[2, 3])  # Rotate in H-W plane
        
        # Add small amount of noise
        if torch.rand(1) < self.prob:
            noise = torch.randn_like(image) * 0.01
            image = torch.clamp(image + noise, 0, 1)
        
        return image


# Modified 3D ResNet for larger input size and binary classification
class ModifiedWideResNet3D(nn.Module):
    """Modified WideResNet3D for larger input and binary classification"""
    
    def __init__(self, input_size=(64, 64, 64), width=2, num_classes=2, dropout_rate=0.3):
        super(ModifiedWideResNet3D, self).__init__()
        
        k = width
        nChannels = [16, 16*k, 32*k, 64*k, 128*k]
        
        # Initial convolution - downsample immediately for large inputs
        self.conv1 = nn.Conv3d(1, nChannels[0], kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(nChannels[0])
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with progressive downsampling
        self.block1 = self._make_layer(nChannels[0], nChannels[1], 2, stride=1, dropout_rate=dropout_rate)
        self.block2 = self._make_layer(nChannels[1], nChannels[2], 2, stride=2, dropout_rate=dropout_rate)
        self.block3 = self._make_layer(nChannels[2], nChannels[3], 2, stride=2, dropout_rate=dropout_rate)
        self.block4 = self._make_layer(nChannels[3], nChannels[4], 2, stride=2, dropout_rate=dropout_rate)
        
        # Final layers
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(nChannels[4], num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, stride, dropout_rate))
        
        for i in range(1, num_blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, 1, dropout_rate))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x), negative_slope=0.01)
        x = self.maxpool(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock3D(nn.Module):
    """Basic 3D residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock3D, self).__init__()
        
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        out = F.leaky_relu(self.bn1(x), negative_slope=0.01)
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), negative_slope=0.01)
        
        if self.dropout:
            out = self.dropout(out)
            
        out = self.conv2(out)
        out += self.shortcut(x)
        
        return out


class Trainer:
    """Training class for 3D ResNet"""
    
    def __init__(self, model, device, save_dir='checkpoints', mixed_precision=False):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.mixed_precision = mixed_precision
        os.makedirs(save_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Initialize mixed precision training
        if self.mixed_precision and device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            print("Mixed precision training enabled")
        else:
            self.scaler = None
            if self.mixed_precision:
                print("Mixed precision requested but CUDA not available")
        
        # Initialize CSV file
        self.csv_path = os.path.join(save_dir, 'training_metrics.csv')
        self.init_csv()
    def init_csv(self):
        """Initialize CSV file with headers"""
        df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
        df.to_csv(self.csv_path, index=False)
    
    def save_metrics_to_csv(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Save metrics for current epoch to CSV"""
        new_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        }
        
        df = pd.DataFrame([new_row])
        df.to_csv(self.csv_path, mode='a', header=False, index=False)
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=1e-4):
        """Train the model"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.8, patience=5)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            
            # Save metrics to CSV
            self.save_metrics_to_csv(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_model.pth')
                print(f'New best model saved with Val Acc: {val_acc:.4f}')
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc, f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.4f}')
        print(f'Training metrics saved to: {self.csv_path}')
        return best_val_acc
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = criterion(output, target)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Standard precision
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'GPU_Mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })
        
        return running_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Mixed precision inference
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'GPU_Mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })
        
        return running_loss / len(val_loader), correct / total
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f'Test Accuracy: {accuracy:.4f}')
        print('\nClassification Report:')
        print(report)
        print('\nConfusion Matrix:')
        print(cm)
        
        return accuracy, report, cm
    
    def save_checkpoint(self, epoch, val_acc, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        return checkpoint['epoch'], checkpoint['val_accuracy']
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, label='Train Loss')
        ax1.plot(epochs, self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, label='Train Acc')
        ax2.plot(epochs, self.val_accuracies, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate curve
        ax3.plot(epochs, self.learning_rates, label='Learning Rate', color='orange')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.show()
        
        # Also save metrics as a final CSV summary
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies,
            'lr': self.learning_rates
        })
        metrics_df.to_csv(os.path.join(self.save_dir, 'final_training_metrics.csv'), index=False)
        print(f"Final metrics summary saved to: {os.path.join(self.save_dir, 'final_training_metrics.csv')}")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': r'D:\Kananat\Data\training_dataset_3D\training_dataset_OA',  # Update this path
        'target_size': (128, 128, 128),      # Resize from 255x255x255 to 64x64x64
        'batch_size': 4,                  # Small batch size due to large images
        'num_epochs': 500,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_workers': 0,                 # Set to 0 for Windows compatibility
        'mixed_precision': True,          # Enable mixed precision training
    }
    
    # Device configuration with optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        print(f'CUDA Version: {torch.version.cuda}')
        
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        # Enable deterministic operations (optional, may reduce performance)
        # torch.backends.cudnn.deterministic = True
    else:
        print('CUDA not available, using CPU')
        config['mixed_precision'] = False
    
    # Data transforms
    train_transform = SimpleTransform(prob=0.5)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = NiftiDataset(
        os.path.join(config['data_dir'], 'train'),
        transform=train_transform,
        target_size=config['target_size']
    )
    
    val_dataset = NiftiDataset(
        os.path.join(config['data_dir'], 'val'),
        transform=None,
        target_size=config['target_size']
    )
    
    test_dataset = NiftiDataset(
        os.path.join(config['data_dir'], 'test'),
        transform=None,
        target_size=config['target_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = ModifiedWideResNet3D(
        input_size=config['target_size'],
        width=2,
        num_classes=2,
        dropout_rate=0.3
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create trainer
    trainer = Trainer(model, device, save_dir='checkpoints', mixed_precision=config['mixed_precision'])
    
    # Train model
    print("Starting training...")
    best_val_acc = trainer.train(
        train_loader, 
        val_loader, 
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Load best model and evaluate
    print("Loading best model for final evaluation...")
    trainer.load_checkpoint('best_model.pth')
    test_acc, test_report, test_cm = trainer.evaluate(test_loader)
    
    # Save final results
    results = {
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'config': config,
        'model_parameters': total_params,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(trainer.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Training completed! Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    # Update the data_dir path before running
    main()