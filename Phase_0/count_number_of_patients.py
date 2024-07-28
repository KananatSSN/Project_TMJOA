import os

# Path to the root directory named "A folder"
data_path = r"C:\Users\acer\Desktop\Back up\raw_Data_and_extra\Data"

count = 0

# Loop through each directory in the data path
for year_folder in os.listdir(data_path):

    # Loop through each directory in the year path
    year_path = os.path.join(data_path, year_folder)
    for patient_folder in os.listdir(year_path):
        
        count += 1

        # Check if the patient folder contains "_Z" and other items
        patient_path = os.path.join(year_path, patient_folder)
        contents = os.listdir(patient_path)
        z_count = contents.count('_Z')
        other_items = len(contents) - z_count
        
        # Print the result for this folder
        if z_count == 1 and other_items == 0:
            #print(f'"{patient_folder}" Pass')
            t = 1
        elif z_count == 0:
            print(f"{patient_path}")
            print(f"{patient_folder} doesn't contain _Z folder")
        else:
            print(f"{patient_path}")
            print(f"{patient_folder} It has {z_count} _Z folders and {other_items} other items. Check required.")

print(count)