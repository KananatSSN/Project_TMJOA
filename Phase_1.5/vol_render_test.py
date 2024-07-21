#exec(open(r"C:\Users\acer\Desktop\Project\Code\Phase_1.5\vol_render_test.py").read())
import time

for i in range(1, 4):
    print(i)
    time.sleep(1)  # Pause for 1 second

def showVolumeRenderingMIP(volumeNode):
    # Get/create volume rendering display node
    volRenLogic = slicer.modules.volumerendering.logic()
    displayNode = volRenLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
    if not displayNode:
        displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
    # Choose MIP volume rendering preset
    displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName("CT-AAA"))
    # Show volume rendering
    displayNode.SetVisibility(True)

volume_path = r"C:\Users\acer\Desktop\Test_input\vol\5450_2016_02_26_L.nii"
masterVolumeNode = slicer.util.loadVolume(volume_path)
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
showVolumeRenderingMIP(volumeNode)
slicer.app.processEvents()

for i in range(1, 4):
    print(i)
    time.sleep(1)  # Pause for 1 second

slicer.mrmlScene.RemoveNode(volumeNode)
slicer.app.processEvents()

volume_path = r"C:\Users\acer\Desktop\Test_output\5450_2016_02_26_L_masked.nii.gz"
masterVolumeNode = slicer.util.loadVolume(volume_path)
volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
showVolumeRenderingMIP(volumeNode)