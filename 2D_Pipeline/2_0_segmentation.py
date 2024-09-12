# This script is use to run the segmentation in 3DSlicer.
# Input : The path to root folder where [root_folder] contains .nii.gz files

# This script should be run in the 3DSlicer python console with the command below
# exec(open(r"C:\Users\acer\Desktop\Project_TMJOA\2D_Pipeline\2_0_segmentation.py").read())

import slicer
from DentalSegmentatorLib import SegmentationWidget, ExportFormat
from pathlib import Path
import os
import shutil

def single_segmentation(filepath, outputfolder):

    file_name = os.path.basename(filepath)
    file_name_without_extension = file_name.split('.')[0]

    print(f"Processing {file_name}")
    
    widget = SegmentationWidget()
    selectedFormats = ExportFormat.NIFTI 
    
    node = slicer.util.loadVolume(filepath)

    widget.inputSelector.setCurrentNode(node)
    widget.applyButton.clicked()
    widget.logic.waitForSegmentationFinished()
    slicer.app.processEvents()
    
    segmentationNode = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))[0]
    segmentation = segmentationNode.GetSegmentation()

    label = f"{file_name_without_extension}_segmented"
    segmentId = "Segment_2" # Segment_[1-5] = ["Maxilla & Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]

    segment = segmentation.GetSegment(segmentId)
    if segment is None:
        print(f"Error: There is no TMJ in {file_name}")
        return
    
    segment.SetName(label)
    singleSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    singleSegmentationNode.SetName(label)
    singleSegmentationNode.GetSegmentation().AddSegment(segment)

    widget.exportSegmentation(singleSegmentationNode, outputfolder, selectedFormats)

    slicer.mrmlScene.RemoveNode(singleSegmentationNode)
    slicer.mrmlScene.RemoveNode(segmentationNode)
    slicer.mrmlScene.RemoveNode(node)
    print(f"Finish processing {file_name}")

###########################################################################################

input_folder = r"C:\Users\acer\Desktop\Data_0"
output_folder = r"C:\Users\acer\Desktop\Data_0\Segmentation"

nii_count = len([filename for filename in os.listdir(input_folder) if filename.endswith('.nii.gz')])
print(f"There are {nii_count} .nii.gz files in the {input_folder}")

progress_count = 0

files = sorted(os.listdir(input_folder), reverse=True)

for filename in files :
    if filename.endswith('.nii.gz'):
        progress_count += 1
        print(f"[Processing {progress_count} out of {nii_count}]")

        file_path = os.path.join(input_folder, filename)
        single_segmentation(file_path, output_folder)

        source = file_path
        destination = r"C:\Users\acer\Desktop\Data_0\Processed"

        shutil.move(source, destination)
        print(f"Moves {filename} from {source} to {destination}")

print("Done")