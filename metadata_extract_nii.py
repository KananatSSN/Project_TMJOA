import nibabel as nib

# Load the .nii file
file_path = r"C:\Users\acer\Desktop\Data_Prep_0\imagesTr\474881_2014-9_L.nii"
nii_image = nib.load(file_path)

# Extract the header information
header = nii_image.header

# Print the metadata
print("Header Information:")
print(header)

# Access specific metadata
print("\nSpecific Metadata:")
print("Data Type:", header.get_data_dtype())
print("Dimension Information:", header.get_data_shape())
print("Voxel Size:", header.get_zooms())