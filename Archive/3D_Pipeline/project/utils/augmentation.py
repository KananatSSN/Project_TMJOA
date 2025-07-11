import torchio as tio
import numpy as np
import torch

class TMJAugmentations:
    """TMJ-specific augmentation pipeline preserving anatomical relationships"""
    
    def __init__(self, training=True):
        self.training = training
        
        if tio is not None:
            # TorchIO augmentations (preferred for medical imaging)
            self.train_transforms = tio.Compose([
                # Spatial transforms with conservative parameters
                tio.RandomAffine(
                    scales=(0.95, 1.05),  # Physiological scaling
                    degrees=(-8, 8),      # Conservative rotation for TMJ
                    translation=(-4, 4),   # 4mm max translation
                    p=0.8
                ),
                
                # Elastic deformation simulating natural variation
                tio.RandomElasticDeformation(
                    num_control_points=(5, 5, 5),
                    max_displacement=(3, 3, 3),  # 3mm max displacement
                    p=0.6
                ),
                
                # Intensity augmentations for scanner variation
                tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5),
                tio.RandomNoise(std=(0, 0.05), p=0.4),
                tio.RandomBiasField(coefficients=0.3, p=0.3),
                
                # Flip with anatomical consideration
                tio.RandomFlip(axes=(0,), p=0.3),  # Sagittal flip only
                
                # Normalize intensity
                tio.RescaleIntensity(out_min_max=(0, 1))
            ])
            
            self.val_transforms = tio.Compose([
                tio.RescaleIntensity(out_min_max=(0, 1))
            ])
            
        else:
            # Fallback to custom augmentations
            print("Using custom augmentations (TorchIO not available)")
            
    def __call__(self, volume):
        if tio is not None:
            # Convert to tensor if needed and ensure correct shape
            if isinstance(volume, np.ndarray):
                volume_tensor = torch.from_numpy(volume).float()
            else:
                volume_tensor = volume.float()
            
            # Ensure 4D tensor (1, D, H, W) for TorchIO
            if volume_tensor.ndim == 3:
                volume_tensor = volume_tensor.unsqueeze(0)  # Add channel dimension
            elif volume_tensor.ndim == 5:
                volume_tensor = volume_tensor.squeeze(0)    # Remove extra dimension
            
            # Create TorchIO Subject
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=volume_tensor)
            )
            
            if self.training:
                transformed = self.train_transforms(subject)
            else:
                transformed = self.val_transforms(subject)
                
            # Return as numpy array to maintain consistency
            result = transformed.image.data.squeeze().numpy()
            
            # Ensure result has channel dimension for consistency
            if result.ndim == 3:
                result = np.expand_dims(result, axis=0)
                
            return result
        else:
            return self.custom_augment(volume)
    
    def custom_augment(self, volume):
        """Fallback custom augmentations"""
        if not self.training:
            return torch.from_numpy(volume) if isinstance(volume, np.ndarray) else volume
            
        # Convert to tensor if needed
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume)
            
        # Simple rotation (around center)
        if torch.rand(1) < 0.5:
            angle = torch.rand(1) * 16 - 8  # ±8 degrees
            # Simplified rotation implementation
            pass
            
        # Add noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(volume) * 0.05
            volume = volume + noise
            
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        return volume

class MixUp3D:
    """3D MixUp augmentation for medical volumes"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam

class CutMix3D:
    """3D CutMix augmentation for medical volumes"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random bounding box
        D, H, W = x.size(2), x.size(3), x.size(4)
        cut_rat = np.sqrt(1. - lam)
        cut_d = int(D * cut_rat)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        
        # Random center point
        cd = np.random.randint(D)
        ch = np.random.randint(H)
        cw = np.random.randint(W)
        
        bbd1 = np.clip(cd - cut_d // 2, 0, D)
        bbd2 = np.clip(cd + cut_d // 2, 0, D)
        bbh1 = np.clip(ch - cut_h // 2, 0, H)
        bbh2 = np.clip(ch + cut_h // 2, 0, H)
        bbw1 = np.clip(cw - cut_w // 2, 0, W)
        bbw2 = np.clip(cw + cut_w // 2, 0, W)
        
        x[:, :, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2] = x[index, :, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbd2 - bbd1) * (bbh2 - bbh1) * (bbw2 - bbw1) / (D * H * W))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam