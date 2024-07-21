# exec(open(r"C:\Users\acer\Desktop\Project_TMJOA\dcm_to_nii.py").read())

dicomDataDir = r"C:\Users\acer\Desktop\Back up\raw_Data_and_extra\Data\57-2014\47-4881 2014-9 L dicom\_Z"  # input folder with DICOM files
loadedNodeIDs = []  # this list will contain the list of all loaded node IDs

from DICOMLib import DICOMUtils
with DICOMUtils.TemporaryDICOMDatabase() as db:
  DICOMUtils.importDicom(dicomDataDir, db)
  patientUIDs = db.patients()
  for patientUID in patientUIDs:
    loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

slicer.app.processEvents()

volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
slicer.mrmlScene.RemoveNode(volumeNode)
slicer.app.processEvents()

# slicer.util.selectModule("DICOM")
# dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser
# # Get the currently selected series ID
# seriesInstanceUIDs = dicomBrowser.dicomTableManager().seriesTable().currentSelection
# seriesInstanceUID = seriesInstanceUIDs[0]  # just use the first series in this example
# # Get the DICOM file paths for this series
# files = slicer.dicomDatabase.filesForSeries(seriesInstanceUID)
# # Show metadata viewer
# metadataViewer = ctk.ctkDICOMObjectListWidget()
# metadataViewer.setFileList(files)
# metadataViewer.show()