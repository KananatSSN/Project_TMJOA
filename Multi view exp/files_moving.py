import os
import shutil
from pathlib import Path
import glob

def organize_images_by_dataset_structure(dataset_path, images_path, output_path):
    """
    Organize PNG images based on the folder structure of NIfTI dataset.
    
    Args:
        dataset_path (str): Path to the original dataset with .nii.gz files
        images_path (str): Path to the folder containing PNG images
        output_path (str): Path where the new organized dataset will be created
    """
    
    # Convert to Path objects for easier manipulation
    dataset_path = Path(dataset_path)
    images_path = Path(images_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    # Iterate through each subset (Train, Val, Test)
    for subset_dir in dataset_path.iterdir():
        if subset_dir.is_dir():
            for class_dir in subset_dir.iterdir():
                if class_dir.is_dir():
                    subset_name = subset_dir.name
                    class_name = class_dir.name
                    print(f"Processing {subset_name} split, class {class_name}")
                    
                    # Create corresponding directory in output
                    output_subset_dir = output_path / subset_name / class_name
                    output_subset_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process each .nii.gz file in the subset
                    for nii_file in class_dir.glob("*.nii.gz"):
                        
                        patient_id = nii_file.name.split('_')[0]  # Get the part before the first underscore

                        # print(f"  Looking for images for {patient_id}...")
                        
                        # Find all PNG images for this patient (recursively)
                        pattern = f"{patient_id}*.png"
                        matching_images = []
                        
                        # Search recursively in the images folder
                        for image_file in images_path.rglob(pattern):
                            matching_images.append(image_file)
                        
                        # Copy matching images to the output directory
                        copied_count = 0
                        for image_file in matching_images:
                            destination = output_subset_dir / image_file.name
                            try:
                                shutil.copy2(image_file, destination)
                                copied_count += 1
                                # print(f"    Copied: {image_file} to {destination}")
                            except Exception as e:
                                print(f"    Error copying {image_file.name}: {e}")

                        print(f"    Found and copied {copied_count} images for patient {patient_id}")
                        
                        # if copied_count == 32:
                        #     # print(f" Success")
                        #     success_count += 1
                        # else:
                        #     print(f" Fail")
    
    print(f"\nDataset organization complete! New dataset created at: {output_path}")
    print(f"Total successful patients with 32 images: {success_count}")
# Example usage
if __name__ == "__main__":
    # Define your paths here
    dataset_path = r"d:\Kananat\Data\training_dataset_3D\training_dataset_genSclerosis"  # Path to your original dataset with .nii.gz files
    images_path = r"d:\Kananat\Data\training_dataset_2D\training_dataset_osteophyte"    # Path to your folder with PNG images
    output_path = r"d:\Kananat\Data\training_dataset_2D\training_dataset_genSclerosis"  # Path where you want the new organized dataset
    
    # Organize the images
    organize_images_by_dataset_structure(dataset_path, images_path, output_path)