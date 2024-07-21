import os

# Path to the root directory named "A folder"
root_path = r"C:\Users\acer\Desktop\Back up\raw_Data_and_extra"

# Loop through each directory in the root path
for folder_name in os.listdir(root_path):

    # Construct the path to this folder
    folder_path = os.path.join(root_path, folder_name)

    # Check if the current path is a directory
    if os.path.isdir(folder_path):

        # List all the contents in this folder
        contents = os.listdir(folder_path)
        
        # Count how many "Z" folders there are
        z_count = contents.count('_Z')
        
        # Determine if there are other items in the folder besides "Z"
        other_items = len(contents) - z_count
        
        # Print the result for this folder
        if z_count == 1 and other_items == 0:
            print(f'"{folder_name}" Pass')
        elif z_count == 1:
            print(f'"{folder_name}" Not pass')
        else:
            print(f'"{folder_name}" does not meet the required criteria. It has {z_count} "_Z" folders and {other_items} other items. Check required.')

