import os
import glob
import shutil

def getUniqueID(files_list):
    unique_id = []
    for file in files_list:
        patient_id = file.split('_')[0]
        if patient_id not in unique_id:
            unique_id.append(patient_id)
    return unique_id

def groupingByID(input_folder):
    for root, dir, file in os.walk(input_folder):
        # print(f"root: {root}")
        # print(f"file: {file}")
        ids = getUniqueID(file)
        for id in ids:
            print(f"Processing : {id}")

            patient_image_folder = os.path.join(root, id)
            os.makedirs(patient_image_folder, exist_ok=True)
            
            images_path = glob.glob(f"{root}\\{id}*")

            for image_path in images_path:
                image_name = os.path.basename(image_path)
                destination = os.path.join(patient_image_folder, image_name)
                #print(image_path)
                # shutil.copy2(image_path, destination)
                print(f"moved {image_path} to {destination}")

if __name__ == "__main__":
    input_folder = r"C:\Users\acer\Desktop\Project_TMJOA\Data\output\training_dataset_OA\test"
    groupingByID(input_folder)