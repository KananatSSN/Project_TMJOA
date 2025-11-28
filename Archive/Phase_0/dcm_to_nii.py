# exec(open(r"C:\Users\kanan\Desktop\Project_TMJOA\Archive\Phase_0\dcm_to_nii.py").read())

####################################################################################
# Name construction function

import os
import pydicom
from DICOMLib import DICOMUtils

def get_dicom_metadata(file_path):
    # Read the DICOM file
    ds = pydicom.dcmread(file_path, force=True)
    # Extract metadata
    metadata = {
        'PatientID': ds.get('PatientID', 'N/A'),
        'StudyDate': ds.get('StudyDate', 'N/A')
    }
    return metadata

def construct_name(folder_path):

    dcm_path = os.path.join(folder_path,r"_Z\SLZ+000.dcm")

    folder_name = os.path.basename(folder_path)

    if 'L' in folder_name:
        LR_tag = 'L'
    elif 'R' in folder_name:
        LR_tag = 'R'
    else:
        LR_tag = 'unknown'

    meta = get_dicom_metadata(dcm_path)

    return f"{meta['PatientID'][:2]}-{meta['PatientID'][2:]}_{meta['StudyDate'][:4]}_{meta['StudyDate'][4:6]}_{meta['StudyDate'][6:8]}_{LR_tag}"

####################################################################################
# Main loop

# data_path = r"d:\Kananat\Data\Sorted_More_data"
# output_folder = r"d:\Kananat\Data\Sorted_More_data_nii"

# for folder_name in os.listdir(data_path):
    
#     patient_folder = os.path.join(data_path,folder_name)

#     if os.path.isdir(patient_folder):

#         print(f"Processing {folder_name}")

#         # Check if _Z folder exist
#         dicomDataDir = os.path.join(patient_folder,r"_Z")
#         if not os.path.isdir(dicomDataDir):
#             print(f"ERROR : There is no _Z folder in {os.path.basename(patient_folder)}")
#             continue

#         # Load dcm to 3dslicer node
#         loadedNodeIDs = []
#         with DICOMUtils.TemporaryDICOMDatabase() as db:
#             DICOMUtils.importDicom(dicomDataDir, db)
#             patientUIDs = db.patients()
#             for patientUID in patientUIDs:
#                 loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
#         volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
#         slicer.app.processEvents()

#         # Save node to .nii.gz
#         output_name = construct_name(patient_folder)
#         output_name = f"{output_name}.nii.gz"
#         output_path = os.path.join(output_folder,output_name)
#         slicer.util.saveNode(volumeNode, output_path)
#         slicer.app.processEvents()

#         # Remove node
#         slicer.mrmlScene.RemoveNode(volumeNode)
#         slicer.app.processEvents()
#         print("Success")

# print("Done")

####################################################################################
# process _z

to_process = [r"d:\Kananat\Data\Sorted_More_data\MSDH 65-32395 L F78 23-1-68\dicom\6532395_20250421122014_Z",
              r"d:\Kananat\Data\Sorted_More_data\MSDH 630010001 L F52 27-5-67\dicom\630010001_20250421123920_Z",
              r"d:\Kananat\Data\Sorted_More_data\MSDH 630010001 R F52 27-5-67\dicom\630010001_20250421123622_Z"]

output_folder = r"d:\Kananat\Data\Sorted_More_data_nii"

for folder in to_process:
    print(f"Processing {folder}")
    loadedNodeIDs = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(folder, db)
        patientUIDs = db.patients()
        for patientUID in patientUIDs:
            loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
    volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    slicer.app.processEvents()

    # Save node to .nii.gz
    dcm_path = os.path.join(folder, "SLZ+000.dcm")

    folder_name = os.path.basename(os.path.dirname(os.path.dirname(folder)))

    if 'L' in folder_name:
        LR_tag = 'L'
    elif 'R' in folder_name:
        LR_tag = 'R'
    else:
        LR_tag = 'unknown'

    meta = get_dicom_metadata(dcm_path)

    output_name = f"{meta['PatientID'][:2]}-{meta['PatientID'][2:]}_{meta['StudyDate'][:4]}_{meta['StudyDate'][4:6]}_{meta['StudyDate'][6:8]}_{LR_tag}"

    output_name = f"{output_name}.nii.gz"
    print(f"saveing {output_name}")
    output_path = os.path.join(output_folder,output_name)
    slicer.util.saveNode(volumeNode, output_path)
    slicer.app.processEvents()

    # Remove node
    slicer.mrmlScene.RemoveNode(volumeNode)
    slicer.app.processEvents()
    print("Success")