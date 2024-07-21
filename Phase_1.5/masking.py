# Run this program in 3dslicer Python Console with the command below
# exec(open(r"C:\Users\acer\Desktop\Project\Code\Phase_1.5\masking.py").read())

import os

def segmentation_masking(volume_path, segmentation_path, output_folder):

    # Load input volume and segmentation
    masterVolumeNode = slicer.util.loadVolume(volume_path)
    segmentationNode = slicer.util.loadSegmentation(segmentation_path)
    slicer.app.processEvents()

    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    # segmentEditorWidget.show() # Uncomment this line to debug thing in the program
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(segmentEditorNode)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)
    slicer.app.processEvents()

    # Set up masking parameters
    segmentEditorWidget.setActiveEffectByName("Mask volume")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("FillValue", "-4000")
    effect.setParameter("Operation", "FILL_OUTSIDE")
    maskedVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Temporary masked volume")
    effect.self().outputVolumeSelector.setCurrentNode(maskedVolume)
    effect.self().onApply()
    slicer.app.processEvents()

    # Set up output directory
    input_file_name = os.path.basename(volume_path)
    input_file_name = os.path.splitext(input_file_name)[0]
    input_file_name = os.path.splitext(input_file_name)[0]
    output_file_name = f"{input_file_name}_masked.nii.gz"
    output_path = os.path.join(output_folder,output_file_name)

    slicer.util.saveNode(maskedVolume, output_path)
    slicer.app.processEvents()

    # Reset 3dslicer memory
    slicer.mrmlScene.RemoveNode(maskedVolume)
    slicer.app.processEvents()
    slicer.mrmlScene.RemoveNode(segmentationNode)
    slicer.app.processEvents()
    slicer.mrmlScene.RemoveNode(masterVolumeNode)
    slicer.app.processEvents()

    print(f"Finish processing {input_file_name}")

###########################################################################################
# Usage sample

# volume_path = r"C:\Users\acer\Desktop\Data_Prep_1\imagesTr\5450_2016_02_26_L.nii"
# segmentation_path = r"C:\Users\acer\Desktop\Data_Prep_1\labelsTr\Output\5450_2016_02_26_L_segmented_filled.nii.gz"
# output_folder = r"C:\Users\acer\Desktop\Data_Prep_2\imagesTr"

# segmentation_masking(volume_path, segmentation_path, output_folder)

###########################################################################################

volume_folder = r"C:\Users\acer\Desktop\Data_Prep_1\imagesTr"
segment_folder = r"C:\Users\acer\Desktop\Data_Prep_1\labelsTr\Output"
output_folder = r"C:\Users\acer\Desktop\Data_Prep_2\imagesTr"

nii_count = len([filename for filename in os.listdir(volume_folder) if filename.endswith('.nii')])
print(f"There are {nii_count} .nii files in the {volume_folder}")

progress_count = 0

files = sorted(os.listdir(volume_folder))

for filename in files :
    if filename.endswith('.nii'):
        progress_count += 1
        print(f"[Processing {progress_count} out of {nii_count}]")

        volume_path = os.path.join(volume_folder, filename)
        segmentation_path = os.path.splitext(filename)[0]
        segmentation_path = f"{segmentation_path}_segmented_filled.nii.gz"
        segmentation_path = os.path.join(segment_folder,segmentation_path)

        if os.path.exists(volume_path) and os.path.exists(segmentation_path):
            segmentation_masking(volume_path, segmentation_path, output_folder)
        else:
            vol_exist = os.path.exists(volume_path)
            seg_exist = os.path.exists(segmentation_path)
            print(f"Error : vol {vol_exist}, seg {seg_exist}")

print("Done")