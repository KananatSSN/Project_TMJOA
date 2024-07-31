import os
import pandas as pd
from pathlib import Path

# Path to your folder
folder_path = r"C:\Users\acer\Desktop\Data_0\Processed"

# Path to your Excel file
excel_file = r"C:\Users\acer\Desktop\Data_0\Classification_ignore_missing.xlsx"

# Column name in Excel file containing file names
column_name = "ID"  # Replace with your actual column name

# Get list of files in the folder and remove the .nii.gz extension
folder_files = set(os.path.splitext(os.path.splitext(f)[0])[0] for f in os.listdir(folder_path) if f.endswith('.nii.gz'))

# Read file names from Excel
df = pd.read_excel(excel_file)
excel_files = set(df[column_name])

# Find files in folder but not in Excel
in_folder_not_excel = folder_files - excel_files

# Find files in Excel but not in folder
in_excel_not_folder = excel_files - folder_files

# Check for perfect 1-1 correspondence
if len(in_folder_not_excel) == 0 and len(in_excel_not_folder) == 0:
    print("Perfect 1-1 correspondence between folder and Excel file.")
else:
    print("Discrepancies found:")
    if in_folder_not_excel:
        print("Files in folder but not in Excel:", in_folder_not_excel)
    if in_excel_not_folder:
        print("Files in Excel but not in folder:", in_excel_not_folder)

# Print summary
print(f"\nTotal files in folder: {len(folder_files)}")
print(f"Total files in Excel: {len(excel_files)}")
print(f"Files in folder but not in Excel: {len(in_folder_not_excel)}")
print(f"Files in Excel but not in folder: {len(in_excel_not_folder)}")