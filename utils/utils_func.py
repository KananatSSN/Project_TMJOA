import nibabel as nib
import os

def check_nifti(file_path):
    """
    Check the dimensions of a 3D image in .nii.gz format
    
    Parameters:
    -----------
    file_path : str
        Path to the .nii.gz file
        
    Returns:
    --------
    dict
        Dictionary containing dimensions, voxel sizes, and other metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if not (file_path.endswith('.nii.gz') or file_path.endswith('.nii')):
        raise ValueError(f"File {file_path} is not a NIfTI file")
    
    # Load the NIfTI image
    img = nib.load(file_path)
    
    # Get the image data
    data = img.get_fdata()
    
    # Get dimensions
    dimensions = data.shape
    
    # Get voxel sizes (in mm)
    voxel_sizes = img.header.get_zooms()
    
    # Get the affine transformation matrix
    affine = img.affine
    
    # Get the datatype
    data_type = img.get_data_dtype()
    
    # Return the information as a dictionary
    return {
        'file_path': file_path,
        'dimensions': dimensions,
        'voxel_sizes': voxel_sizes,
        'n_dimensions': len(dimensions),
        'n_voxels': data.size,
        'data_type': str(data_type),
        'affine': affine
    }

def define_background(input_path, output_path, range=[-250, 1500], background_value=-250):
    """
    Define the background of a NIfTI image by setting values outside a specified range to a background value.
    
    Parameters:
    -----------
    input_path : str
        Path to the input NIfTI file
    output_path : str
        Path to save the modified NIfTI file
    range : tuple
        Range of values to keep in the image (default: (-250, 1500))
    background_value : int or float
        Value to set for the background (default: 0)
        
    Returns:
    --------
    None
    """
    # Load the NIfTI image
    img = nib.load(input_path)
    
    # Get the image data
    data = img.get_fdata()
    
    # Set values outside the specified range to the background value
    data[(data < range[0]) | (data > range[1])] = background_value
    
    # Create a new NIfTI image with the modified data
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    
    # Save the new image
    nib.save(new_img, output_path)
