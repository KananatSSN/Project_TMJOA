import os
import nibabel as nib
import numpy as np

def quickfix(input_folder, output_folder):
    BG_VALUE = -2000
    files = os.listdir(input_folder)
    founded_files = len(files)
    print(f"Found {founded_files} files")

    os.makedirs(output_folder, exist_ok=True)

    progress = 0
    for file in files:
        
        if not file.endswith('.nii.gz'):
            print("Not .nii.gz, skipped")
            continue

        progress += 1
        print(f"Progress {progress}/{founded_files}")

        image_path = os.path.join(input_folder, file)
        image = nib.load(image_path)

        image_data = image.get_fdata()

        image_data = np.round(image_data, decimals=1)
        image_data = np.clip(image_data, a_min = BG_VALUE, a_max = None)

        output_path = os.path.join(output_folder, file)

        new_img = nib.Nifti1Image(image_data, image.affine, image.header)
        nib.save(new_img, output_path)

if __name__ == "__main__":
    input_folder = r"C:\Users\acer\Desktop\Project_TMJOA\Data\Open access data\Follow_up_rmDoubleBg"
    output_folder = r"C:\Users\acer\Desktop\Project_TMJOA\Data\Open access data\Follow_up_fixed"
    quickfix(input_folder, output_folder)