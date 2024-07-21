# exec(open(r"C:\Users\acer\Desktop\Project_TMJOA\Phase_0\dcm_to_nii.py").read())

####################################################################################
# Name construction function

import os
import pandas as pd
import pydicom

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

    return f"{meta['PatientID']}_{meta['StudyDate'][:4]}_{meta['StudyDate'][4:6]}_{meta['StudyDate'][6:8]}_{LR_tag}"

####################################################################################
# Main loop
from DICOMLib import DICOMUtils

data_path = r"C:\Users\acer\Desktop\Test_input\Data"
output_folder = r"C:\Users\acer\Desktop\Test_output"
in_data_folder = os.listdir(data_path)

for folder_name in in_data_folder:
    year_folder = os.path.join(data_path,folder_name)
    in_year_folder = os.listdir(year_folder)

    patient_count = len(in_year_folder)
    print(f"There are {patient_count} .nii files in the {os.path.basename(year_folder)}")
    progress_count = 0

    for folder_name in in_year_folder:
        
        # Progress bar
        progress_count += 1
        print(f"[Processing {progress_count} out of {patient_count}]")

        # Working directory
        patient_folder = os.path.join(year_folder,folder_name)

        # Load dcm to 3dslicer node
        dicomDataDir = os.path.join(patient_folder,r"_Z")
        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDataDir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.app.processEvents()

        # Save node to .nii.gz
        output_name = construct_name(patient_folder)
        output_name = f"{output_name}.nii.gz"
        output_path = os.path.join(output_folder,output_name)
        slicer.util.saveNode(volumeNode, output_path)
        slicer.app.processEvents()

        # Remove node
        slicer.mrmlScene.RemoveNode(volumeNode)
        slicer.app.processEvents()

print("Done")