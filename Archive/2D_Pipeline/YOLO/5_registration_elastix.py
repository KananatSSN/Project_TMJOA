# Run this program in 3dslicer Python Console with the command below
# exec(open(r"C:\Users\kanan\Desktop\Project_TMJOA\2D_Pipeline\YOLO\5_registration.py").read())

# Clear node
slicer.mrmlScene.Clear(0)

import os
import Elastix
import timeit

# Get the Elastix logic
elastixLogic = Elastix.ElastixLogic()

# Path to fixed reference image - CHANGE THIS PATH
fixed_path = r"D:\Kananat\Data\47-4881 L 2014_registration_referance.nii.gz"
fixed_mask_path = r"D:\Kananat\Data\1_Augmented_segmentation\47-4881 L 2014_segmented_augmented.nii.gz"

# Path to folder containing moving volumes - CHANGE THIS PATH
moving_folder = r"D:\Kananat\Data\2_Masked"
moving_mask_folder = r"D:\Kananat\Data\1_Augmented_segmentation"

# Path to output folder - CHANGE THIS PATH
output_folder = r"D:\Kananat\Data\3_Registed"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Specify the parameter set ID
parameter_set_id = "default-rigid"

files = os.listdir(moving_folder)
number_of_files = len(files)

# Process each file in the moving folder
count = 0
for filename in files[0:3]:
    if filename.endswith(".nii.gz"):

        count = count + 1
        start_time = timeit.default_timer()

        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]  # Remove .nii.gz

        print(f"Processing {count} of {number_of_files}: {filename}")

        # Load fixed volume
        fixed_volume = slicer.util.loadVolume(fixed_path)
        print(f"Load fixed volume : {fixed_path}")

        # Load fixed volume mask
        fixed_mask_volume = slicer.util.loadSegmentation(fixed_mask_path)
        fixed_mask_labelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(fixed_mask_volume, fixed_mask_labelmap, fixed_volume)
        print(f"Load fixed mask volume : {fixed_mask_path}")

        # Load moving volume
        moving_path = os.path.join(moving_folder, filename)
        moving_volume = slicer.util.loadVolume(moving_path)
        print(f"Load moving volume : {moving_path}")

        # Load moving volume mask
        base_base_name = base_name.split("_")[0]
        moving_mask_filename = f"{base_base_name}_segmented_augmented.nii.gz"
        moving_mask_path = os.path.join(moving_mask_folder, moving_mask_filename)
        moving_mask_volume = slicer.util.loadSegmentation(moving_mask_path)
        moving_mask_labelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(moving_mask_volume, moving_mask_labelmap, moving_volume)
        print(f"Load moving mask volume : {moving_mask_path}")
        
        # Construct output filename
        output_filename = f"{base_name}_registered.nii.gz"
        output_path = os.path.join(output_folder, output_filename)
        
        # Create output volume node
        output_volume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', f"{base_base_name}_registered")
        
        # Load parameter preset for registration
        preset = elastixLogic.getPresetByID(parameter_set_id)
        parameterFilenames = preset.getParameterFiles()
        print(f"Load parameter preset from : {parameterFilenames}")
        
        # Run registration with the parameter set ID
        elastixLogic.registerVolumes(
            fixedVolumeNode=fixed_volume, 
            movingVolumeNode=moving_volume, 
            outputVolumeNode=output_volume,
            fixedVolumeMaskNode=fixed_mask_volume,
            movingVolumeMaskNode=moving_mask_volume,
            parameterFilenames=parameterFilenames
        )
        
        # Save the registered volume
        slicer.util.saveNode(output_volume, output_path)
        print(f"Saved registered volume: {output_path}")
        
        # Clean up nodes to free memory
        slicer.mrmlScene.RemoveNode(moving_volume)
        slicer.mrmlScene.RemoveNode(output_volume)
        slicer.mrmlScene.RemoveNode(fixed_volume)
        slicer.mrmlScene.Clear(0)

        stop_time = timeit.default_timer()
        execution_time = stop_time - start_time
        print(f"Processing time : {str(execution_time)}")

print("Registration complete for all volumes!")