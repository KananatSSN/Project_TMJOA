import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from typing import List, Callable


class CBCTRandAugment:
    """
    RandAugment implementation for preprocessed CBCT volumes.
    Input: torch.Tensor of shape [1, n, n, n], dtype=float32, range=[0,1]
    Works with any cubic volume size.
    """
    
    def __init__(self, n: int = 2, m: int = 6):
        """
        Args:
            n: Number of transformations to apply sequentially
            m: Magnitude of transformations (0-10 scale)
        """
        self.n = n
        self.m = m
        
        # Define augmentation operations (no preprocessing needed)
        self.operations: List[Callable] = [
            self._random_rotation,
            self._random_flip,
            self._random_translation,
            self._random_noise,
            self._random_gamma,
            self._random_contrast,
            self._random_gaussian_blur,
            self._elastic_deformation,
        ]
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply RandAugment to preprocessed CBCT volume
        
        Args:
            image: Tensor of shape [1, n, n, n], range [0,1]
        Returns:
            Augmented tensor of same shape and range
        """
        # Validate input
        assert len(image.shape) == 4, f"Expected 4D tensor, got {len(image.shape)}D"
        assert image.shape[0] == 1, f"Expected channel dimension = 1, got {image.shape[0]}"
        assert image.dtype == torch.float32, f"Expected float32, got {image.dtype}"
        
        # Select N random operations
        selected_ops = np.random.choice(self.operations, self.n, replace=False)
        
        # Apply operations sequentially
        for op in selected_ops:
            image = op(image)
            # Ensure values stay in [0,1] range
            image = torch.clamp(image, 0.0, 1.0)
        
        return image
    
    def _scale_magnitude(self, max_val: float) -> float:
        """Scale magnitude from [0,10] to [0, max_val]"""
        return (self.m / 10.0) * max_val
    
    def _random_rotation(self, image: torch.Tensor) -> torch.Tensor:
        """Random 3D rotation with magnitude-scaled angle"""
        max_angle = 30  # Maximum 30 degrees
        angle = self._scale_magnitude(max_angle)
        
        # Random rotation around random axis
        axis = np.random.randint(0, 3)  # 0=D, 1=H, 2=W
        angle_rad = np.random.uniform(-angle, angle) * np.pi / 180
        
        # Create rotation matrix for the chosen axis
        if axis == 0:  # Rotate around depth axis (H-W plane)
            dims = [2, 3]
        elif axis == 1:  # Rotate around height axis (D-W plane)
            dims = [1, 3]
        else:  # Rotate around width axis (D-H plane)
            dims = [1, 2]
        
        # Apply rotation using torch.rot90 for 90-degree increments or affine for arbitrary angles
        if abs(angle) < 5:  # Small angles - use identity
            return image
        elif abs(angle) % 90 < 5:  # Close to 90-degree multiples
            k = int(round(angle / 90)) % 4
            return torch.rot90(image, k=k, dims=dims)
        else:
            # For arbitrary angles, use affine transformation
            return self._apply_affine_rotation(image, angle_rad, axis)
    
    def _apply_affine_rotation(self, image: torch.Tensor, angle: float, axis: int) -> torch.Tensor:
        """Apply arbitrary angle rotation using affine transformation"""
        # Convert to numpy for scipy operations
        img_np = image.squeeze(0).numpy()
        
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        center = np.array(img_np.shape) / 2
        
        # Create rotation matrix
        if axis == 0:  # Rotate in H-W plane
            matrix = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])
        elif axis == 1:  # Rotate in D-W plane
            matrix = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        else:  # Rotate in D-H plane
            matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        
        # Apply rotation
        coords = np.mgrid[0:img_np.shape[0], 0:img_np.shape[1], 0:img_np.shape[2]]
        coords = coords.reshape(3, -1)
        coords = coords - center.reshape(3, 1)
        rotated_coords = matrix @ coords
        rotated_coords = rotated_coords + center.reshape(3, 1)
        
        # Interpolate
        rotated_img = map_coordinates(img_np, rotated_coords.reshape(3, *img_np.shape), 
                                    order=1, mode='constant', cval=0.0)
        
        return torch.tensor(rotated_img, dtype=torch.float32).unsqueeze(0)
    
    def _random_flip(self, image: torch.Tensor) -> torch.Tensor:
        """Random flip along one or more axes"""
        # Randomly choose axes to flip (excluding channel dimension)
        axes_to_flip = []
        for dim in [1, 2, 3]:  # D, H, W dimensions
            if torch.rand(1) < 0.5:
                axes_to_flip.append(dim)
        
        for dim in axes_to_flip:
            image = torch.flip(image, dims=[dim])
        
        return image
    
    def _random_translation(self, image: torch.Tensor) -> torch.Tensor:
        """Random translation with magnitude-scaled displacement"""
        max_translation = 0.2  # 20% of dimension size
        translation = self._scale_magnitude(max_translation)
        
        # Random translation for each axis
        shifts = []
        for dim_size in image.shape[1:]:  # Skip channel dimension
            max_shift = int(translation * dim_size)
            shift = np.random.randint(-max_shift, max_shift + 1)
            shifts.append(shift)
        
        # Apply translation using roll
        for i, shift in enumerate(shifts):
            if shift != 0:
                image = torch.roll(image, shifts=shift, dims=i+1)
        
        return image
    
    def _random_noise(self, image: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with magnitude-scaled standard deviation"""
        max_std = 0.1  # Maximum noise std
        noise_std = self._scale_magnitude(max_std)
        
        if noise_std > 0:
            noise = torch.randn_like(image) * noise_std
            image = image + noise
        
        return image
    
    def _random_gamma(self, image: torch.Tensor) -> torch.Tensor:
        """Gamma correction with magnitude-scaled gamma value"""
        max_gamma_change = 0.4  # ±0.2 around 1.0
        gamma_change = self._scale_magnitude(max_gamma_change)
        gamma = 1.0 + np.random.uniform(-gamma_change/2, gamma_change/2)
        
        # Apply gamma correction
        image = torch.pow(image, gamma)
        
        return image
    
    def _random_contrast(self, image: torch.Tensor) -> torch.Tensor:
        """Adjust contrast with magnitude-scaled factor"""
        max_contrast_change = 0.4  # ±20% contrast change
        contrast_change = self._scale_magnitude(max_contrast_change)
        contrast_factor = 1.0 + np.random.uniform(-contrast_change/2, contrast_change/2)
        
        # Apply contrast adjustment around mean
        mean_val = torch.mean(image)
        image = (image - mean_val) * contrast_factor + mean_val
        
        return image
    
    def _random_gaussian_blur(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur with magnitude-scaled sigma"""
        max_sigma = 2.0  # Maximum blur sigma
        sigma = self._scale_magnitude(max_sigma)
        
        if sigma > 0.1:  # Only apply if sigma is significant
            # Convert to numpy for scipy gaussian filter
            img_np = image.squeeze(0).numpy()
            blurred = gaussian_filter(img_np, sigma=sigma)
            image = torch.tensor(blurred, dtype=torch.float32).unsqueeze(0)
        
        return image
    
    def _elastic_deformation(self, image: torch.Tensor) -> torch.Tensor:
        """Apply elastic deformation with magnitude-scaled displacement"""
        max_displacement = 10  # Maximum displacement in pixels
        displacement = self._scale_magnitude(max_displacement)
        
        if displacement < 1:  # Skip if displacement too small
            return image
        
        # Create random displacement field
        shape = image.shape[1:]  # Remove channel dimension
        dx = np.random.uniform(-displacement, displacement, shape)
        dy = np.random.uniform(-displacement, displacement, shape)
        dz = np.random.uniform(-displacement, displacement, shape)
        
        # Smooth the displacement field
        sigma = displacement / 3
        dx = gaussian_filter(dx, sigma=sigma)
        dy = gaussian_filter(dy, sigma=sigma)
        dz = gaussian_filter(dz, sigma=sigma)
        
        # Create coordinate grids (ensure float type)
        coords = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
        coords[0] += dx
        coords[1] += dy
        coords[2] += dz
        
        # Apply deformation
        img_np = image.squeeze(0).numpy()
        deformed = map_coordinates(img_np, coords, order=1, mode='reflect')
        
        return torch.tensor(deformed, dtype=torch.float32).unsqueeze(0)


# Example usage:
if __name__ == "__main__":
    # Test with different sizes
    for size in [64, 128, 256]:
        sample_volume = torch.rand(1, size, size, size, dtype=torch.float32)
        
        # Initialize augmentation
        augment = CBCTRandAugment(n=2, m=6)
        
        # Apply augmentation
        augmented_volume = augment(sample_volume)
        
        print(f"Size {size}: Original shape: {sample_volume.shape}")
        print(f"Size {size}: Augmented shape: {augmented_volume.shape}")
        print(f"Size {size}: Original range: [{sample_volume.min():.3f}, {sample_volume.max():.3f}]")
        print(f"Size {size}: Augmented range: [{augmented_volume.min():.3f}, {augmented_volume.max():.3f}]")
        print()