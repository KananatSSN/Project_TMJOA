import nibabel as nib
import os

def check_nifti_dimensions(file_path):
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