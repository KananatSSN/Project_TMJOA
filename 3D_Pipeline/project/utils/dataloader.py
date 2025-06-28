from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import numpy as np
import nibabel as nib

from utils.augmentation import TMJAugmentations, MixUp3D, CutMix3D

tmj_augment_train = TMJAugmentations(training=True)
tmj_augment_val = TMJAugmentations(training=False)
mixup = MixUp3D(alpha=0.2)
cutmix = CutMix3D(alpha=1.0)

class TMJDataset(Dataset):
    """TMJ 3D volume dataset for pre-split train/val/test data"""
    
    def __init__(self, data_path, split='train', transform=None, target_size=(64, 64, 64)):
        self.data_path = Path(data_path)
        self.split = split  # 'train', 'val', or 'test'
        self.transform = transform
        self.target_size = target_size
        
        # Load data paths and labels from the specified split
        self.data_list = []
        self.labels = []
        self.patient_ids = []
        
        # Path to the specific split folder
        split_path = self.data_path / split
        if not split_path.exists():
            raise ValueError(f"Split folder {split_path} does not exist!")
        
        # Load Class 0 files
        class0_path = split_path / '0'
        if class0_path.exists():
            for file_path in class0_path.glob('*'):
                if self._is_valid_medical_file(file_path):
                    self.data_list.append(file_path)
                    self.labels.append(0)
                    # Extract patient ID from filename (modify as needed)
                    patient_id = self._extract_patient_id(file_path.name)
                    self.patient_ids.append(patient_id)
        
        # Load Class 1 files
        class1_path = split_path / '1'
        if class1_path.exists():
            for file_path in class1_path.glob('*'):
                if self._is_valid_medical_file(file_path):
                    self.data_list.append(file_path)
                    self.labels.append(1)
                    patient_id = self._extract_patient_id(file_path.name)
                    self.patient_ids.append(patient_id)
        
        print(f"Loaded {split} split with {len(self.data_list)} samples:")
        print(f"  Class 0: {sum(1 for l in self.labels if l == 0)} samples")
        print(f"  Class 1: {sum(1 for l in self.labels if l == 1)} samples")
        
        if len(self.data_list) == 0:
            print(f"Warning: No valid files found in {split_path}")
            print("Expected structure:")
            print(f"  {data_path}/")
            print(f"    ├── train/")
            print(f"    │   ├── 0/  (class 0 files)")
            print(f"    │   └── 1/  (class 1 files)")
            print(f"    ├── val/")
            print(f"    │   ├── 0/  (class 0 files)")
            print(f"    │   └── 1/  (class 1 files)")
            print(f"    └── test/")
            print(f"        ├── 0/  (class 0 files)")
            print(f"        └── 1/  (class 1 files)")
        
    def _is_valid_medical_file(self, file_path):
        """Check if file is a valid medical image format"""
        valid_extensions = ['.nii', '.nii.gz', '.npy', '.npz', '.dcm']
        return any(str(file_path).lower().endswith(ext) for ext in valid_extensions)
    
    def _extract_patient_id(self, filename):
        """Extract patient ID from filename - modify based on your naming convention"""
        # Example: 'patient_001_scan.nii.gz' -> 'patient_001'
        # Modify this based on your actual filename pattern
        return filename.split('_')[0] if '_' in filename else filename.split('.')[0]
    
    def _load_volume(self, file_path):
        """Load 3D volume from various medical formats"""
        file_path = str(file_path)
        
        if file_path.endswith(('.nii', '.nii.gz')):
            # NIfTI format
            img = nib.load(file_path)
            volume = img.get_fdata()
        elif file_path.endswith('.npy'):
            # NumPy format
            volume = np.load(file_path)
        elif file_path.endswith('.npz'):
            # Compressed NumPy format
            data = np.load(file_path)
            volume = data['volume'] if 'volume' in data.files else data[data.files[0]]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return volume.astype(np.float32)
    
    def _preprocess_volume(self, volume):
        """Preprocess volume to target size and normalize"""
        # Handle different input shapes
        if volume.ndim == 4:
            volume = volume[..., 0]  # Take first channel if 4D
        
        # Resize to target size
        volume = self._resize_volume(volume, self.target_size)
        
        # Normalize intensity
        volume = self._normalize_intensity(volume)
        
        # Add channel dimension
        volume = np.expand_dims(volume, axis=0)
        
        return volume
    
    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation"""
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        
        # Use order=1 (linear) for medical images to preserve intensities
        resized_volume = zoom(volume, zoom_factors, order=1)
        
        return resized_volume
    
    def _normalize_intensity(self, volume):
        """Normalize intensity values"""
        # Clip outliers (optional - adjust percentiles as needed)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Min-max normalization
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        return volume
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # Load volume
        volume = self._load_volume(self.data_list[idx])
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        
        # Preprocess (this adds channel dimension)
        volume = self._preprocess_volume(volume)
        
        # Apply transforms (handle potential shape changes)
        if self.transform:
            volume = self.transform(volume)
        
        # Convert to tensor and ensure correct shape
        if not isinstance(volume, torch.Tensor):
            volume = torch.from_numpy(volume).float()
        
        # Ensure volume has correct shape: (C, D, H, W)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # Add channel dimension if missing
        elif volume.ndim == 5:
            volume = volume.squeeze(0)    # Remove extra dimension if present
        
        return volume, torch.tensor(label, dtype=torch.long), patient_id

def create_dataloaders(data_path, config, use_balanced_sampling=True):
    """Create train, validation, and test dataloaders for pre-split data"""
    
    # Create datasets for each split
    train_dataset = TMJDataset(
        data_path=data_path,
        split='train',
        transform=tmj_augment_train,
        target_size=config['input_size']
    )
    
    val_dataset = TMJDataset(
        data_path=data_path,
        split='val',
        transform=tmj_augment_val,
        target_size=config['input_size']
    )
    
    test_dataset = TMJDataset(
        data_path=data_path,
        split='test',
        transform=tmj_augment_val,
        target_size=config['input_size']
    )
    
    # Create balanced sampler for training if requested
    train_sampler = None
    if use_balanced_sampling and len(train_dataset) > 0:
        train_sampler = get_balanced_sampler(train_dataset.labels)
        print("Using balanced sampling for training data")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=0,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def get_balanced_sampler(labels):
    """Create balanced sampler for handling class imbalance"""
    from torch.utils.data import WeightedRandomSampler
    
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler