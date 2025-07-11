import os
import numpy as np
import nibabel as nib
from PIL import Image
import argparse
from pathlib import Path

def calculate_nonbackground_ratio(slice_2d, background_threshold=0):
    """
    Calculate the ratio of non-background to background pixels in a 2D slice.
    
    Args:
        slice_2d: 2D numpy array representing the slice
        background_threshold: Threshold to determine background pixels (default: 0)
    
    Returns:
        float: Ratio of non-background to background pixels
    """
    # Count background pixels (pixels <= threshold)
    background_pixels = np.sum(slice_2d <= background_threshold)
    
    # Count non-background pixels
    non_background_pixels = np.sum(slice_2d > background_threshold)
    
    # Calculate ratio (avoid division by zero)
    if background_pixels == 0:
        return float('inf')  # All pixels are non-background
    
    return non_background_pixels / background_pixels

def normalize_slice_for_saving(slice_2d, method='minmax'):
    """
    Normalize a 2D slice for saving as image.
    
    Args:
        slice_2d: 2D numpy array
        method: Normalization method ('minmax' or 'percentile')
    
    Returns:
        numpy array: Normalized slice ready for saving
    """
    if method == 'minmax':
        # Min-max normalization to 0-255
        slice_min = slice_2d.min()
        slice_max = slice_2d.max()
        if slice_max > slice_min:
            normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(slice_2d, dtype=np.uint8)
    elif method == 'percentile':
        # Percentile-based normalization (robust to outliers)
        p2, p98 = np.percentile(slice_2d, (2, 98))
        normalized = np.clip((slice_2d - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Method must be 'minmax' or 'percentile'")
    
    return normalized

def extract_2d_slices(nifti_file_path, output_dir, ratio_threshold, 
                     background_threshold=0, dimension=0, 
                     normalization_method='minmax', image_format='png'):
    """
    Extract 2D slices from a NIfTI file based on non-background to background ratio.
    
    Args:
        nifti_file_path: Path to the .nii.gz file
        output_dir: Directory to save extracted slices
        ratio_threshold: Minimum ratio of non-background to background pixels
        background_threshold: Threshold to determine background pixels
        dimension: Dimension to extract from (0, 1, or 2)
        normalization_method: Method for normalizing pixel values
        image_format: Output image format ('png', 'jpg', 'tiff')
    """
    try:
        # Load the NIfTI file
        img = nib.load(nifti_file_path)
        data = img.get_fdata()
        
        print(f"Processing {nifti_file_path}")
        print(f"Image shape: {data.shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract slices along the specified dimension
        num_slices = data.shape[dimension]
        saved_slices = 0
        
        for i in range(num_slices):
            # Extract 2D slice based on dimension
            if dimension == 0:
                slice_2d = data[i, :, :]
            elif dimension == 1:
                slice_2d = data[:, i, :]
            elif dimension == 2:
                slice_2d = data[:, :, i]
            else:
                raise ValueError("Dimension must be 0, 1, or 2")
            
            # Calculate non-background to background ratio
            ratio = calculate_nonbackground_ratio(slice_2d, background_threshold)
            
            # Save slice if ratio exceeds threshold
            if ratio > ratio_threshold:
                # Normalize slice for saving
                normalized_slice = normalize_slice_for_saving(slice_2d, normalization_method)
                
                # Create filename
                filename = f"slice_{i:04d}.{image_format}"
                filepath = os.path.join(output_dir, filename)
                
                # Save as image
                Image.fromarray(normalized_slice).save(filepath)
                saved_slices += 1
        
        print(f"Saved {saved_slices} slices from {num_slices} total slices")
        
    except Exception as e:
        print(f"Error processing {nifti_file_path}: {str(e)}")

def process_dataset(dataset_path, output_base_path, ratio_threshold, background_threshold=0, 
                   dimension=0, normalization_method='minmax', image_format='png'):
    """
    Process the entire dataset folder structure.
    
    Args:
        dataset_path: Path to the dataset folder containing subfolders 0 and 1
        output_base_path: Base path for output folders
        ratio_threshold: Minimum ratio of non-background to background pixels
        background_threshold: Threshold to determine background pixels
        dimension: Dimension to extract from (0, 1, or 2)
        normalization_method: Method for normalizing pixel values
        image_format: Output image format
    """
    dataset_path = Path(dataset_path)
    output_base_path = Path(output_base_path)
    
    # Create base output directory if it doesn't exist
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Process subfolders 0 and 1
    for subfolder in ['0', '1']:
        subfolder_path = dataset_path / subfolder
        
        if not subfolder_path.exists():
            print(f"Subfolder {subfolder} not found in {dataset_path}")
            continue
        
        print(f"\nProcessing subfolder: {subfolder}")
        
        # Find all .nii.gz files in the subfolder
        nifti_files = list(subfolder_path.glob("*.nii.gz"))
        
        if not nifti_files:
            print(f"No .nii.gz files found in {subfolder_path}")
            continue
        
        for nifti_file in nifti_files:
            # Create output directory with the same name as the file (without extension)
            file_stem = nifti_file.name.replace('.nii.gz', '')
            output_dir = output_base_path / subfolder / file_stem
            
            # Extract 2D slices
            extract_2d_slices(
                nifti_file_path=str(nifti_file),
                output_dir=str(output_dir),
                ratio_threshold=ratio_threshold,
                background_threshold=background_threshold,
                dimension=dimension,
                normalization_method=normalization_method,
                image_format=image_format
            )

def main():
    parser = argparse.ArgumentParser(description='Extract 2D slices from NIfTI files')
    parser.add_argument('dataset_path', help='Path to the dataset folder')
    parser.add_argument('output_path', help='Path to the output folder')
    parser.add_argument('ratio_threshold', type=float, 
                       help='Minimum non-background to background pixel ratio')
    parser.add_argument('--background_threshold', type=float, default=0,
                       help='Threshold to determine background pixels (default: 0)')
    parser.add_argument('--dimension', type=int, default=0, choices=[0, 1, 2],
                       help='Dimension to extract from (default: 0)')
    parser.add_argument('--normalization', choices=['minmax', 'percentile'], 
                       default='minmax', help='Normalization method (default: minmax)')
    parser.add_argument('--format', choices=['png', 'jpg', 'tiff'], 
                       default='png', help='Output image format (default: png)')
    
    args = parser.parse_args()
    
    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        output_base_path=args.output_path,
        ratio_threshold=args.ratio_threshold,
        background_threshold=args.background_threshold,
        dimension=args.dimension,
        normalization_method=args.normalization,
        image_format=args.format
    )

if __name__ == "__main__":
    # Example usage without command line arguments
    # Uncomment and modify these lines to run directly
    
    dataset_path = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Baseline_dataset\3D\test"
    output_path = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Baseline_dataset\3D\test_patientwise"
    ratio_threshold = 0.1  # Adjust this value as needed
    process_dataset(dataset_path, output_path, ratio_threshold, normalization_method='minmax')
    
    # Or run with command line arguments
    # main()