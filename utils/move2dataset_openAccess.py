import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def organize_dataset(source_folder, csv_file, target_dataset_folder, col_name=['Patient', 'Baseline_Health_status'], 
                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                    file_extension=None):
    """
    Organize files from source folder to train/val/test structure based on CSV labels.
    
    Args:
        source_folder: Path to folder containing files (001, 002, etc.)
        csv_file: Path to CSV file with Patient and Baseline_Health_status columns
        target_dataset_folder: Path to target dataset folder
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.2)
        test_ratio: Ratio for test set (default 0.1)
        file_extension: File extension to look for (e.g., '.jpg', '.png'). If None, will auto-detect
    """
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Check required columns
    if col_name[0] not in df.columns or col_name[1] not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"CSV must contain {col_name[0]} and {col_name[1]} columns")
    
    print(f"Found {len(df)} labeled samples")
    print(f"Label distribution:\n{df[col_name[1]].value_counts()}")
    
    # Create target directory structure
    splits = ['train', 'val', 'test']
    labels = ['0', '1']
    
    for split in splits:
        for label in labels:
            os.makedirs(os.path.join(target_dataset_folder, split, label), exist_ok=True)
    
    # Get all files in source folder
    if not os.path.exists(source_folder):
        raise ValueError(f"Source folder does not exist: {source_folder}")
    
    # Auto-detect file extension if not provided
    if file_extension is None:
        all_files = os.listdir(source_folder)
        if all_files:
            # Find the most common extension
            extensions = {}
            for f in all_files:
                if '.' in f:
                    ext = os.path.splitext(f)[1].lower()
                    extensions[ext] = extensions.get(ext, 0) + 1
            if extensions:
                file_extension = max(extensions, key=extensions.get)
                print(f"Auto-detected file extension: {file_extension}")
            else:
                file_extension = ""  # No extension
    
    # Find available files and match with labels
    available_files = []
    missing_patients = []
    
    # Get all files in source folder
    all_files = os.listdir(source_folder)
    
    for _, row in df.iterrows():
        # Convert patient ID to integer first to remove any decimal points, then pad with zeros
        patient_id = str(int(row[col_name[0]])).zfill(3)  # Pad with zeros to get 001, 002, etc.
        label = int(row[col_name[1]])
        
        # Find ALL files that match this patient ID
        matching_files = []
        
        # Try different file naming patterns
        possible_patterns = [
            f"{patient_id}",  # 001
            f"{int(row[col_name[0]])}"  # 1 (without zero padding)
        ]
        
        for filename in all_files:
            file_base = os.path.splitext(filename)[0]  # Remove extension
            
            # Check if this file matches any of the patient patterns
            for pattern in possible_patterns:
                if file_base == pattern or file_base.startswith(pattern + '_') or file_base.startswith(pattern + '-'):
                    file_path = os.path.join(source_folder, filename)
                    matching_files.append({
                        'patient_id': patient_id,
                        'file_path': file_path,
                        'filename': filename,
                        'label': label
                    })
                    break
        
        if matching_files:
            available_files.extend(matching_files)
        else:
            missing_patients.append(patient_id)
    
    print(f"Found {len(available_files)} files out of {len(df)} labeled patients")
    if missing_patients:
        print(f"Missing files for patients: {missing_patients[:10]}{'...' if len(missing_patients) > 10 else ''}")
    
    # Show files per patient distribution
    files_per_patient = pd.DataFrame(available_files).groupby('patient_id').size()
    print(f"Files per patient - Min: {files_per_patient.min()}, Max: {files_per_patient.max()}, Mean: {files_per_patient.mean():.1f}")
    
    if len(available_files) == 0:
        raise ValueError("No matching files found. Please check file naming convention and extension.")
    
    # Convert to DataFrame for easier manipulation
    files_df = pd.DataFrame(available_files)
    
    # Group by patient_id to ensure all files from same patient go to same split
    patient_groups = files_df.groupby('patient_id').first().reset_index()  # One row per patient for splitting
    
    # Stratified split based on patients (not individual files)
    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        patient_groups, 
        test_size=(val_ratio + test_ratio), 
        stratify=patient_groups['label'],
        random_state=42
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, 
        test_size=(1 - val_size), 
        stratify=temp_patients['label'],
        random_state=42
    )
    
    # Now get all files for each patient group
    train_files = files_df[files_df['patient_id'].isin(train_patients['patient_id'])]
    val_files = files_df[files_df['patient_id'].isin(val_patients['patient_id'])]
    test_files = files_df[files_df['patient_id'].isin(test_patients['patient_id'])]
    
    # Copy files to appropriate directories
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    copy_count = 0
    for split_name, split_df in splits_data.items():
        unique_patients = split_df['patient_id'].nunique()
        total_files = len(split_df)
        print(f"\n{split_name.capitalize()} set: {unique_patients} patients, {total_files} files")
        
        label_counts = split_df['label'].value_counts().sort_index()
        for label in label_counts.index:
            patients_with_label = split_df[split_df['label'] == label]['patient_id'].nunique()
            files_with_label = label_counts[label]
            print(f"  Label {label}: {patients_with_label} patients, {files_with_label} files")
        
        for _, row in split_df.iterrows():
            source_path = row['file_path']
            target_dir = os.path.join(target_dataset_folder, split_name, str(row['label']))
            target_path = os.path.join(target_dir, row['filename'])
            
            # Copy file
            shutil.copy2(source_path, target_path)
            copy_count += 1
    
    print(f"\nSuccessfully copied {copy_count} files to {target_dataset_folder}")
    print(f"Dataset structure created:")
    train_patients_0 = train_files[train_files['label'] == 0]['patient_id'].nunique()
    train_files_0 = len(train_files[train_files['label'] == 0])
    train_patients_1 = train_files[train_files['label'] == 1]['patient_id'].nunique()
    train_files_1 = len(train_files[train_files['label'] == 1])
    
    val_patients_0 = val_files[val_files['label'] == 0]['patient_id'].nunique()
    val_files_0 = len(val_files[val_files['label'] == 0])
    val_patients_1 = val_files[val_files['label'] == 1]['patient_id'].nunique()
    val_files_1 = len(val_files[val_files['label'] == 1])
    
    test_patients_0 = test_files[test_files['label'] == 0]['patient_id'].nunique()
    test_files_0 = len(test_files[test_files['label'] == 0])
    test_patients_1 = test_files[test_files['label'] == 1]['patient_id'].nunique()
    test_files_1 = len(test_files[test_files['label'] == 1])
    
    print(f"├── train/")
    print(f"│   ├── 0/ ({train_patients_0} patients, {train_files_0} files)")
    print(f"│   └── 1/ ({train_patients_1} patients, {train_files_1} files)")
    print(f"├── val/")
    print(f"│   ├── 0/ ({val_patients_0} patients, {val_files_0} files)")
    print(f"│   └── 1/ ({val_patients_1} patients, {val_files_1} files)")
    print(f"└── test/")
    print(f"    ├── 0/ ({test_patients_0} patients, {test_files_0} files)")
    print(f"    └── 1/ ({test_patients_1} patients, {test_files_1} files)")

# Example usage
if __name__ == "__main__":
    # Configure these paths according to your setup
    SOURCE_FOLDER = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Preprocessed_Baseline\Preprocessed"  # Folder containing 001, 002, etc. files
    CSV_FILE = r"d:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline_label.csv"       # Your CSV file with labels
    TARGET_DATASET = r"D:\Kananat\Data\raw_Data_and_extra\Open access data\Baseline\Baseline_dataset\3D"   # Target folder for train/val/test structure
    
    # Optional: specify file extension (e.g., '.jpg', '.png', '.npy')
    # If None, the script will try to auto-detect
    FILE_EXTENSION = ".nii.gz"  # or ".jpg", ".png", etc.
    COL_NAME = ['Patient', 'Baseline_Health_status']  # Columns to check in CSV
    
    try:
        organize_dataset(
            source_folder=SOURCE_FOLDER,
            csv_file=CSV_FILE,
            target_dataset_folder=TARGET_DATASET,
            col_name=COL_NAME,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            file_extension=FILE_EXTENSION
        )
    except Exception as e:
        print(f"Error: {e}")