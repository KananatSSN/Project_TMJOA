import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def process_nifti(input_path, output_path, min_val, max_val, target_size=(255, 255, 255)):
    """
    Threshold voxels outside [min_val, max_val] and resize to target dimensions
    
    Parameters:
    - input_path: path to input .nii.gz file
    - output_path: path to save output .nii.gz file
    - min_val, max_val: range bounds [a, b]
    - target_size: target dimensions (254, 254, 254)
    - background_value: value for background voxels
    """
    # Load the NIfTI image
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Step 1: Apply thresholding
    data = np.clip(data, min_val, max_val)
    
    # Step 2: Calculate zoom factors for resizing
    current_shape = data.shape
    zoom_factors = [target_size[i] / current_shape[i] for i in range(3)]
    
    # Step 3: Resize the data
    resized_data = zoom(data, zoom_factors, order=1)  # order=1 for linear interpolation
    
    # Step 4: Update the affine matrix to account for the new voxel size
    new_affine = img.affine.copy()
    # Scale the affine matrix by the inverse of zoom factors
    for i in range(3):
        new_affine[:3, i] = new_affine[:3, i] / zoom_factors[i]
    
    # Step 5: Create and save new image
    new_img = nib.Nifti1Image(resized_data.astype(np.float32), new_affine, img.header)
    nib.save(new_img, output_path)
    
    print(f"Original shape: {current_shape} -> New shape: {resized_data.shape}")

def process_folder(input_folder, output_folder, min_val, max_val, target_size=(255, 255, 255)):
    """Process all .nii.gz files in a folder"""
    import os
    
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, filename)

            out_filename = filename.rsplit('_', 1)[0]
            out_filename = f"{out_filename}_preprocessed.nii.gz"

            output_path = os.path.join(output_folder, out_filename)
            
            try:
                process_nifti(input_path, output_path, min_val, max_val, target_size)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Entire folder
input_folder = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Preprocessed_Baseline\Cropped"  # Replace with your input folder path
output_folder = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Preprocessed_Baseline\Preprocessed"  # Replace with your output folder path
process_folder(input_folder, output_folder, min_val=-700, max_val=2700)