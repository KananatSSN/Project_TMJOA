import nibabel as nib
import os

nifti_path = r"C:\Users\acer\Desktop\Project_TMJOA\Data\output\training_dataset_OA\val\1\66-20679 L_adjustedBG.nii.gz"

nii_img = nib.load(nifti_path)
img_data = nii_img.get_fdata()
affine = nii_img.affine
header = nii_img.header

print(f"Loaded NIfTI image: {os.path.basename(nifti_path)}")
print(f"Shape: {img_data.shape}")
print(f"Value range: {img_data.min():.2f} to {img_data.max():.2f}")