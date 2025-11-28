# exec(open(r"C:\Users\kanan\Desktop\Project_TMJOA\Check\3d_volume_check.py").read())
# This script is use to check the 3d volume in 'input_folder' and save the problemetic files's name in 'output_file'
import os, sys, glob
import slicer
from slicer.ScriptedLoadableModule import *

# Define the folder containing the .nii files
input_folder = r"D:\Kananat\Data\Last0\2_Masked"
output_file = rf"{input_folder}\check_result.txt"

with open(output_file, 'w') as log_file:
    log_file.write('')

pattern = rf"{input_folder}\*.nii.gz"
files = glob.glob(pattern)

number_of_files = len(files)

print(f"There are {number_of_files} .nii.gz files in the {input_folder}")

progress = 0

# Load the text file for writing if needed
with open(output_file, 'w') as log_file:
    # Loop over each file in the directory
    for filename in os.listdir(input_folder):
        progress += 1
        print(f"[Processing {progress} out of {number_of_files}]")

        if filename.endswith('.nii.gz'):
            file_path = os.path.join(input_folder, filename)
            # Load the volume
            loadedVolume = slicer.util.loadVolume(file_path)
            
            # Create a volume rendering of the loaded image
            volumeRenderingLogic = slicer.modules.volumerendering.logic()
            displayNode = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(loadedVolume)
            displayNode.GetVolumePropertyNode().Copy(volumeRenderingLogic.GetPresetByName('CT-AAA'))
            slicer.app.layoutManager().resetThreeDViews()
            displayNode.SetVisibility(True)
            slicer.app.processEvents()

            # Wait for user input to continue or log the file
            while True:
                user_input = input(f"{filename}\n[y/n] : ").lower()
                slicer.app.processEvents()
                if user_input == 'y':
                    slicer.app.processEvents()
                    break
                elif user_input == 'n':
                    log_file.write(filename + '\n')
                    #log_file.flush()
                    print(f"Logged {filename}")
                    slicer.app.processEvents()
                    break
                elif user_input == 'stop':
                    print(f"Stopping the program")
                    slicer.app.processEvents()
                    sys.exit()
                else:
                    print("Invalid input, please type 'y' or 'n'.")
                slicer.app.processEvents()

            slicer.mrmlScene.RemoveNode(loadedVolume)

print("Processing complete.")