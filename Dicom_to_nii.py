import pydicom
import os
import numpy as np
import nibabel as nib

def dicom_folder_to_nifti(dicom_folder, output_file):
    # Retrieve all DICOM files
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    # Sort files to maintain the correct order of slices
    dicom_files.sort(key=lambda x: pydicom.dcmread(x, force=True).InstanceNumber)

    # Read and stack slices
    slices = [pydicom.dcmread(f, force=True).pixel_array for f in dicom_files]
    slices = np.stack(slices, axis=-1)  # Adjust axis if necessary based on your DICOM file orientation

    # Get reference DICOM file for metadata
    ref_dicom = pydicom.dcmread(dicom_files[0], force=True)
    # Define the affine transformation matrix
    affine = np.eye(4)  # Customize this based on actual metadata
    # Example to adjust spacing and slice thickness:
    # affine[0,0] = ref_dicom.PixelSpacing[0]
    # affine[1,1] = ref_dicom.PixelSpacing[1]
    # affine[2,2] = ref_dicom.SliceThickness

    # Create and save NIfTI file
    nifti_img = nib.Nifti1Image(slices, affine)
    nib.save(nifti_img, output_file)
    print(f'Created {output_file}')

# Usage example
root_dir = r"C:\Users\acer\Desktop\Project\Data\57-2014\47-4881 2014-9 L dicom"
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        output_nii_path = os.path.join(root_dir, f"{folder_name}.nii")
        dicom_folder_to_nifti(folder_path, output_nii_path)
