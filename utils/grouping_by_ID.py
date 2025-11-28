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

def groupingByID(input_folder, output_folder):
    class_names = os.listdir(input_folder)
    for class_name in class_names:
        images_folder = os.path.join(input_folder, class_name)
        ids = getUniqueID(os.listdir(images_folder))

        for id in ids:
            print(f"Processing : {id} from class {class_name}")

            patient_image_folder = os.path.join(output_folder, class_name, id)
            os.makedirs(patient_image_folder, exist_ok=True)
            
            images_path = glob.glob(f"{images_folder}\\{id}*")

            for image_path in images_path:
                image_name = os.path.basename(image_path)
                destination = os.path.join(patient_image_folder, image_name)
                shutil.copy2(image_path, destination)
            print(f"moved {len(images_path)} to {patient_image_folder}")

if __name__ == "__main__":
    subset = ['erosion_multiview', 'flattening_multiview', 'genSclerosis_multiview', 'OA_multiview', 'osteophyte_multiview', 'subCyst_multiview']
    for subset in subset:
        print(f"Processing subset: {subset}")
        input_folder = rf"d:\Kananat\Data\training_dataset_2D\Multiview\{subset}\val"
        if os.path.isdir(input_folder):    
            output_folder = rf"d:\Kananat\Data\training_dataset_2D\Multiview\{subset}\val_grouped"
            os.makedirs(output_folder, exist_ok=True)
            groupingByID(input_folder, output_folder)
        else:
            print(f"Directory {input_folder} does not exist.")