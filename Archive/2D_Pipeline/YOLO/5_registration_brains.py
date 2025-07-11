# Run this program in 3dslicer Python Console with the command below
# exec(open(r"C:\Users\kanan\Desktop\Project_TMJOA\2D_Pipeline\YOLO\5_registration_brains.py").read())

import os
import timeit

slicer.mrmlScene.Clear(0)

# Get the BRAINSFit module
BRAINSFitCLI = slicer.modules.brainsfit

# CONFIGURE THESE PATHS
fixed_volume_path = r"D:\Kananat\Data\47-4881 L 2014_registration_referance.nii.gz"  # Fixed reference image
fixed_mask_path = r"D:\Kananat\Data\1_Augmented_segmentation\47-4881 L 2014_segmented_augmented.nii.gz"      # Fixed image mask
moving_folder = r"D:\Kananat\Data\2_Masked"          # Folder containing moving images
moving_mask_folder = r"D:\Kananat\Data\1_Augmented_segmentation"       # Folder containing moving masks
output_folder = r"D:\Kananat\Data\3_Registed"                  # Output folder

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load fixed volume and mask once (these remain the same for all registrations)
fixed_volume = slicer.util.loadVolume(fixed_volume_path)
fixed_mask = slicer.util.loadSegmentation(fixed_mask_path)
fixed_mask_labelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', f"ref_labelmap")
slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(fixed_mask, fixed_mask_labelmap, fixed_volume)

# Get list of moving volumes
moving_files = [f for f in os.listdir(moving_folder) if f.endswith('.nii.gz')]
number_of_files = len(moving_files)

# Process each moving volume
count = 0
for moving_file in moving_files:

    count = count + 1
    start_time = timeit.default_timer()

    print(f"Processing {count} of {number_of_files}: {moving_file}")

    # Construct paths
    moving_volume_path = os.path.join(moving_folder, moving_file)
    base_name = os.path.splitext(os.path.splitext(moving_file)[0])[0]
    
    # Construct mask filename - assuming similar naming convention
    base_base_name = base_name.split("_")[0]
    moving_mask_file = f"{base_base_name}_segmented_augmented.nii.gz"  # Adjust naming pattern if needed
    moving_mask_path = os.path.join(moving_mask_folder, moving_mask_file)
    # print(f"Moving vol : {moving_volume_path}")
    # print(f"Moving mask : {moving_mask_path}")
    
    # Check if moving mask exists
    if not os.path.exists(moving_mask_path):
        print(f"Warning: Mask not found for {moving_file}. Skipping registration.")
        continue
    
    # Construct output paths
    output_volume_path = os.path.join(output_folder, f"{base_base_name}_registered.nii.gz")
    # print(f"Output vol : {output_volume_path}")
    
    if os.path.exists(output_volume_path):
        print(f"Already existed, Skipped: {output_volume_path}")
        continue
    
    # Load moving volume and mask
    moving_volume = slicer.util.loadVolume(moving_volume_path)
    moving_mask = slicer.util.loadSegmentation(moving_mask_path)
    moving_mask_labelmap = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', f"{base_base_name}_labelmap")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(moving_mask, moving_mask_labelmap, moving_volume)

    # Create output node
    registered_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{base_base_name}_registered")
    
    # Create parameters for BRAINSFit
    parameters = {}
    parameters["fixedVolume"] = fixed_volume.GetID()
    parameters["movingVolume"] = moving_volume.GetID()
    parameters["outputVolume"] = registered_volume.GetID()
    parameters["maskProcessingMode"] = "ROI"
    parameters["fixedBinaryVolume"] = fixed_mask_labelmap.GetID()
    parameters["movingBinaryVolume"] = moving_mask_labelmap.GetID()
    parameters["useRigid"] = True
    parameters["useAffine"] = True
    parameters["samplingPercentage"] = 0.2
    parameters["numberOfIterations"] = 2000
    parameters["minimumStepLength"] = 0.002
    parameters["splineGridSize"] = [14, 10, 12]
    parameters["initializeTransformMode"] = "useCenterOfHeadAlign"
    parameters["backgroundFillValue"] = -4000 
    
    # Run the registration
    cliNode = slicer.cli.runSync(BRAINSFitCLI, None, parameters)
    
    # Check for errors
    if cliNode.GetStatus() & cliNode.ErrorsMask:
        print(f"Error in BRAINSFit registration for {moving_file}: {cliNode.GetErrorText()}")
        # Clean up before continuing to next file
        slicer.mrmlScene.RemoveNode(cliNode)
        slicer.mrmlScene.RemoveNode(moving_volume)
        slicer.mrmlScene.RemoveNode(moving_mask)
        slicer.mrmlScene.RemoveNode(moving_mask_labelmap)
        slicer.mrmlScene.RemoveNode(registered_volume)
        continue
    
    print(f"Registration completed successfully")
    
    # Save the results
    slicer.util.saveNode(registered_volume, output_volume_path)
    
    print(f"Saved registered volume to: {output_volume_path}")
    
    # Clean up nodes to free memory
    slicer.mrmlScene.RemoveNode(cliNode)
    slicer.mrmlScene.RemoveNode(moving_volume)
    slicer.mrmlScene.RemoveNode(moving_mask)
    slicer.mrmlScene.RemoveNode(moving_mask_labelmap)
    slicer.mrmlScene.RemoveNode(registered_volume)

    stop_time = timeit.default_timer()
    execution_time = stop_time - start_time
    print(f"Processing time : {str(execution_time)}")

# Clean up fixed volume and mask at the end
slicer.mrmlScene.RemoveNode(fixed_volume)
slicer.mrmlScene.RemoveNode(fixed_mask)
slicer.mrmlScene.RemoveNode(fixed_mask_labelmap)

slicer.mrmlScene.Clear(0)
print("Batch registration complete!")