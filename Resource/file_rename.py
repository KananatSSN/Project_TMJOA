import os
def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            # Insert hyphen between 2nd and 3rd character
            new_filename = filename[:2] + '-' + filename[2:]
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Replace this with the path to your folder
folder_path = r"D:\Kananat\Augmented_segmentation_dim1_expand10px"

rename_files(folder_path)