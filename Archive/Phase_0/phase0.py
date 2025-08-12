import os
import pandas as pd
import pydicom
import math
from DICOMLib import DICOMUtils

class Phase0:
    def __init__(self):
        """Initialize Phase0 class for DICOM to NIfTI conversion and label matching"""
        pass
    
    # DICOM metadata methods
    def get_dicom_metadata(self, file_path, extended=False):
        """
        Extract metadata from a DICOM file
        
        Args:
            file_path: Path to the DICOM file
            extended: If True, extract additional metadata fields
        
        Returns:
            Dictionary containing DICOM metadata
        """
        # Read the DICOM file
        ds = pydicom.dcmread(file_path, force=True)
        
        # Basic metadata
        metadata = {
            'PatientID': ds.get('PatientID', 'N/A'),
            'StudyDate': ds.get('StudyDate', 'N/A')
        }
        
        # Extended metadata if requested
        if extended:
            metadata.update({
                'Modality': ds.get('Modality', 'N/A'),
                'Manufacturer': ds.get('Manufacturer', 'N/A'),
                'InstitutionName': ds.get('InstitutionName', 'N/A'),
                'BodyPartExamined': ds.get('BodyPartExamined', 'N/A'),
                'PatientAge': ds.get('PatientAge', 'N/A'),
                'PatientSex': ds.get('PatientSex', 'N/A'),
                'StudyDescription': ds.get('StudyDescription', 'N/A'),
                'SeriesDescription': ds.get('SeriesDescription', 'N/A'),
                'ProtocolName': ds.get('ProtocolName', 'N/A')
            })
            
        return metadata
    
    def process_dicom_folder(self, folder_path):
        """
        Process all DICOM files in a folder and extract metadata
        
        Args:
            folder_path: Path to folder containing DICOM files
            
        Returns:
            DataFrame containing metadata for all DICOM files
        """
        data = []
        
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.dcm'):
                file_path = os.path.join(folder_path, filename)
                metadata = self.get_dicom_metadata(file_path, extended=True)
                metadata['Filename'] = filename
                data.append(metadata)
                
        # Create a DataFrame from the data
        return pd.DataFrame(data)
    
    # DICOM to NIfTI conversion methods
    def construct_name(self, folder_path):
        """
        Construct a filename based on DICOM metadata
        
        Args:
            folder_path: Path to folder containing DICOM files
            
        Returns:
            Constructed filename string
        """
        dcm_path = os.path.join(folder_path, r"_Z\SLZ+000.dcm")
        folder_name = os.path.basename(folder_path)
        
        if 'L' in folder_name:
            LR_tag = 'L'
        elif 'R' in folder_name:
            LR_tag = 'R'
        else:
            LR_tag = 'unknown'
            
        meta = self.get_dicom_metadata(dcm_path)
        
        return f"{meta['PatientID']}_{meta['StudyDate'][:4]}_{meta['StudyDate'][4:6]}_{meta['StudyDate'][6:8]}_{LR_tag}"
    
    def convert_dcm_to_nii(self, data_path, output_folder):
        """
        Convert DICOM files to NIfTI format
        
        Args:
            data_path: Path to root folder containing DICOM files
            output_folder: Path to output folder for NIfTI files
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        in_data_folder = os.listdir(data_path)
        
        for folder_name in in_data_folder:
            year_folder = os.path.join(data_path, folder_name)
            in_year_folder = os.listdir(year_folder)
            
            patient_count = len(in_year_folder)
            print(f"There are {patient_count} .nii files in the {os.path.basename(year_folder)}")
            progress_count = 0
            
            for folder_name in in_year_folder:
                # Progress bar
                progress_count += 1
                print(f"[Processing {progress_count} out of {patient_count}]")
                
                # Working directory
                patient_folder = os.path.join(year_folder, folder_name)
                
                # Check if _Z folder exists
                dicomDataDir = os.path.join(patient_folder, r"_Z")
                if not os.path.isdir(dicomDataDir):
                    print(f"ERROR : There is no _Z folder in {os.path.basename(patient_folder)}")
                    continue
                
                # Load dcm to 3dslicer node
                loadedNodeIDs = []
                with DICOMUtils.TemporaryDICOMDatabase() as db:
                    DICOMUtils.importDicom(dicomDataDir, db)
                    patientUIDs = db.patients()
                    for patientUID in patientUIDs:
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
                volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                slicer.app.processEvents()
                
                # Save node to .nii.gz
                output_name = self.construct_name(patient_folder)
                output_name = f"{output_name}.nii.gz"
                output_path = os.path.join(output_folder, output_name)
                slicer.util.saveNode(volumeNode, output_path)
                slicer.app.processEvents()
                
                # Remove node
                slicer.mrmlScene.RemoveNode(volumeNode)
                slicer.app.processEvents()
                print("Success")
                
        print("Done")
    
    # Label matching methods
    def rename(self, filename):
        """
        Extract patient ID and side information from filename
        
        Args:
            filename: NIfTI filename
            
        Returns:
            String with extracted information
        """
        parts = filename.split('_')
        return f"{parts[0][:2]}-{parts[0][2:]} {parts[4]}"
    
    def find_duplicates_in_list_of_lists(self, lists):
        """
        Find duplicate entries in a list of lists
        
        Args:
            lists: List of lists to check for duplicates
            
        Returns:
            List of duplicate entries
        """
        seen = set()
        duplicates = set()
        for sublist in lists:
            tuple_version = tuple(sublist)  # Convert list to tuple to make it hashable
            if tuple_version in seen:
                duplicates.add(tuple_version)
            else:
                seen.add(tuple_version)
        return [list(dup) for dup in duplicates]  # Convert tuples back to lists
    
    def reconstruct_file_name_from_file_list(self, file_list_0):
        """
        Reconstruct NIfTI filename from file list entry
        
        Args:
            file_list_0: List containing file information
            
        Returns:
            Reconstructed filename
        """
        ID = file_list_0[0].split(' ')[0]
        LR = file_list_0[0].split(' ')[1]
        return f"{ID[:2]}{ID[3:]}_{file_list_0[1][0]}_{file_list_0[1][1]}_{file_list_0[1][2]}_{LR}.nii.gz"
    
    def reconstruct_file_name_from_label_list(self, label_list_0):
        """
        Reconstruct filename from label list entry
        
        Args:
            label_list_0: List containing label information
            
        Returns:
            Reconstructed filename
        """
        label_list_0 = [str(item) for item in label_list_0]
        if label_list_0[1] == 'nan':
            label_list_0 = [label_list_0[0]]
        return f"{' '.join(label_list_0)}.nii.gz"
    
    def rename_files_in_folder(self, folder_path, old_name, new_name):
        """
        Rename a file in a folder
        
        Args:
            folder_path: Path to folder containing file
            old_name: Original filename
            new_name: New filename
        """
        old_file = os.path.join(folder_path, old_name)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
    
    def match_labels_with_nii(self, folder_path, label_xlsx_path):
        """
        Match NIfTI files with labels from Excel file
        
        Args:
            folder_path: Path to folder containing NIfTI files
            label_xlsx_path: Path to Excel file containing labels
        """
        # List all files in the directory
        file_names = os.listdir(folder_path)
        
        # Create a DataFrame with file names
        df = pd.DataFrame(file_names, columns=['nii_file'])
        df['nii_file'] = df['nii_file'].str.split('.').str[0]
        df['date'] = df['nii_file'].str.split('_').str[1:4]
        
        # Apply rename function to each row
        df['nii_file'] = df['nii_file'].apply(lambda x: self.rename(x))
        
        # Convert to list
        file_list = df.values.tolist()
        
        # Read the Excel file with labels
        excel_df = pd.read_excel(label_xlsx_path)
        excel_df = excel_df[["ID", "Date"]]
        label_list = excel_df.values.tolist()
        
        # Check for duplicates
        duplicates = self.find_duplicates_in_list_of_lists(label_list)
        if duplicates:
            print("Warning: Duplicates found in label list:", duplicates)
        
        # Process each patient in the label list
        for patient in label_list:
            patient_Id = patient[0]
            
            # Count the number of scans this patient took
            nii_of_this_patient = []
            for nii in file_list:  
                if nii[0] == patient_Id:
                    nii_of_this_patient.append(nii)
            
            if len(nii_of_this_patient) == 0:
                print(f"{patient_Id} does not have any scan")
                print("Debug : missing .nii file **********************************************************")
                print("\n")
                continue
            
            # Output in case this patient took only 1 scan
            if len(nii_of_this_patient) == 1 and math.isnan(float(str(patient[1]).split(' ')[0])):
                print("\n")
                print(nii_of_this_patient[0])
                print(patient)
                old_name = self.reconstruct_file_name_from_file_list(nii_of_this_patient[0])
                new_name = self.reconstruct_file_name_from_label_list(patient)
                self.rename_files_in_folder(folder_path, old_name, new_name)
                print(old_name)
                print(new_name)
                continue
            
            if len(nii_of_this_patient) == 1:
                scan_year = str(patient[1]).split(' ')[0]
                if int(nii_of_this_patient[0][1][0]) == int(scan_year):
                    print("\n")
                    print(nii_of_this_patient[0])
                    print(patient)
                    old_name = self.reconstruct_file_name_from_file_list(nii_of_this_patient[0])
                    new_name = self.reconstruct_file_name_from_label_list(patient)
                    self.rename_files_in_folder(folder_path, old_name, new_name)
                    print(old_name)
                    print(new_name)
                    continue
                elif int(nii_of_this_patient[0][1][0]) != int(scan_year):
                    print(f"{patient_Id} does not have any scan from year {scan_year}")
                    print(f"This patient {patient}")
                    print(f"nii files with this patient ID {nii_of_this_patient}")
                    print("Debug : missing .nii file **********************************************************")
                    print("\n")
                    continue
            
            # Process patients with multiple scans in different years/months/days
            # Count the number of scans in this year
            scan_year = str(patient[1]).split(' ')[0]
            nii_of_this_patient_this_year = []
            for nii in nii_of_this_patient:
                if nii[1][0] == scan_year:
                    nii_of_this_patient_this_year.append(nii)
            
            # Output in case there is only 1 scan in this year
            if len(nii_of_this_patient_this_year) == 1:
                print("\n")
                print(nii_of_this_patient_this_year[0])
                print(patient)
                old_name = self.reconstruct_file_name_from_file_list(nii_of_this_patient_this_year[0])
                new_name = self.reconstruct_file_name_from_label_list(patient)
                self.rename_files_in_folder(folder_path, old_name, new_name)
                print(old_name)
                print(new_name)
                continue
                
            # Count the number of scans in this month
            scan_month = str(patient[1]).split(' ')[1]
            nii_of_this_patient_this_year_and_month = []
            for nii in nii_of_this_patient_this_year:
                if nii[1][1] == scan_month:
                    nii_of_this_patient_this_year_and_month.append(nii)
            
            # Output in case there is only 1 scan in this month
            if len(nii_of_this_patient_this_year_and_month) == 1:
                print("\n")
                print(nii_of_this_patient_this_year_and_month[0])
                print(patient)
                old_name = self.reconstruct_file_name_from_file_list(nii_of_this_patient_this_year_and_month[0])
                new_name = self.reconstruct_file_name_from_label_list(patient)
                self.rename_files_in_folder(folder_path, old_name, new_name)
                print(old_name)
                print(new_name)
                continue
            
            # Count the number of scans in this day
            scan_day = str(patient[1]).split(' ')[2]
            nii_of_this_patient_this_year_and_month_and_day = []
            for nii in nii_of_this_patient_this_year_and_month:
                if nii[1][2] == scan_day:
                    nii_of_this_patient_this_year_and_month_and_day.append(nii)
            
            # Output in case there is only 1 scan in this day
            if len(nii_of_this_patient_this_year_and_month_and_day) == 1:
                print("\n")
                print(nii_of_this_patient_this_year_and_month_and_day[0])
                print(patient)
                old_name = self.reconstruct_file_name_from_file_list(nii_of_this_patient_this_year_and_month_and_day[0])
                new_name = self.reconstruct_file_name_from_label_list(patient)
                self.rename_files_in_folder(folder_path, old_name, new_name)
                print(old_name)
                print(new_name)
                continue
