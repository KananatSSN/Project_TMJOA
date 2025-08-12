import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def organize_dataset(source_folder, csv_file, target_dataset_folder, 
                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                    file_extension=None, col_name=['Patient', 'Baseline_Health_status']):
    """
    Organize files from source folder to train/val/test structure based on CSV labels.
    
    Args:
        source_folder: Path to folder containing files (001, 002, etc.)
        csv_file: Path to CSV file with patient IDs and labels
        target_dataset_folder: Path to target dataset folder
        train_ratio: Ratio for training set (default 0.7)
        val_ratio: Ratio for validation set (default 0.2)
        test_ratio: Ratio for test set (default 0.1)
        file_extension: File extension to look for (e.g., '.jpg', '.png'). If None, will auto-detect
        col_name: List of column names [patient_id_column, label_column] (default ['Patient', 'Baseline_Health_status'])
    """
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Get column names from parameter
    patient_col, label_col = col_name
    
    # Check required columns
    if patient_col not in df.columns or label_col not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError(f"CSV must contain '{patient_col}' and '{label_col}' columns")
    
    print(f"Found {len(df)} labeled samples")
    print(f"Label distribution:\n{df[label_col].value_counts()}")
    
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
        # Handle both numeric and string patient IDs
        raw_patient_id = row[patient_col]
        
        if pd.isna(raw_patient_id):
            continue  # Skip rows with missing patient IDs
            
        # Convert to string and clean up, but keep original format for substring matching
        if isinstance(raw_patient_id, (int, float)):
            # For numeric IDs, we still want the clean string version for grouping
            patient_id = str(int(float(raw_patient_id)))
            search_pattern = patient_id
            base_patient_id = patient_id  # For numeric IDs, base is the same
        else:
            # For string IDs, use as-is
            patient_id = str(raw_patient_id).strip()
            search_pattern = patient_id
            
            # Extract base patient ID (everything before the first space)
            # For '47-4881 L 2014', this gives '47-4881'
            base_patient_id = patient_id.split()[0] if ' ' in patient_id else patient_id
        
        label = int(row[label_col])
        
        # Find ALL files that contain this patient ID as substring
        matching_files = []
        
        for filename in all_files:
            # Check if patient ID is a substring of the filename
            if search_pattern in filename:
                file_path = os.path.join(source_folder, filename)
                matching_files.append({
                    'patient_id': patient_id,          # Full ID for file tracking
                    'base_patient_id': base_patient_id, # Base ID for grouping splits
                    'file_path': file_path,
                    'filename': filename,
                    'label': label
                })
        
        if matching_files:
            available_files.extend(matching_files)
        else:
            missing_patients.append(patient_id)
    
    print(f"Found {len(available_files)} files out of {len(df)} labeled patients")
    if missing_patients:
        print(f"Missing files for patients: {missing_patients[:10]}{'...' if len(missing_patients) > 10 else ''}")
    
    # Show files per patient distribution
    files_per_patient = pd.DataFrame(available_files).groupby('base_patient_id').size()
    unique_base_patients = len(files_per_patient)
    print(f"Unique base patients: {unique_base_patients}")
    print(f"Files per base patient - Min: {files_per_patient.min()}, Max: {files_per_patient.max()}, Mean: {files_per_patient.mean():.1f}")
    
    if len(available_files) == 0:
        raise ValueError("No matching files found. Please check file naming convention and extension.")
    
    # Convert to DataFrame for easier manipulation
    files_df = pd.DataFrame(available_files)
    
    # Group by base_patient_id to ensure all files from same base patient go to same split
    # Note: Files from same base patient can have different labels (e.g., L vs R, different years)
    base_patient_groups = files_df.groupby('base_patient_id').agg({
        'patient_id': 'first'  # Keep one example of full patient ID for reference
    }).reset_index()
    
    # For stratification, we need to assign a label to each base patient
    # We'll use the most common label for that base patient, or label 0 if tied
    base_patient_labels = files_df.groupby('base_patient_id')['label'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    ).reset_index()
    base_patient_labels.columns = ['base_patient_id', 'dominant_label']
    
    # Merge to get the dominant label for stratification
    base_patient_groups = base_patient_groups.merge(base_patient_labels, on='base_patient_id')
    
    # Show label distribution within base patients
    multi_label_patients = files_df.groupby('base_patient_id')['label'].nunique()
    mixed_label_count = (multi_label_patients > 1).sum()
    if mixed_label_count > 0:
        print(f"Info: {mixed_label_count} base patients have multiple labels (e.g., L/R, different years)")
        print("This is expected and all files from same base patient will stay in same split.")
    
    # Stratified split based on base patients using dominant label
    # First split: train vs (val + test)
    train_patients, temp_patients = train_test_split(
        base_patient_groups, 
        test_size=(val_ratio + test_ratio), 
        stratify=base_patient_groups['dominant_label'],
        random_state=42
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, 
        test_size=(1 - val_size), 
        stratify=temp_patients['dominant_label'],
        random_state=42
    )
    
    # Now get all files for each base patient group
    train_files = files_df[files_df['base_patient_id'].isin(train_patients['base_patient_id'])]
    val_files = files_df[files_df['base_patient_id'].isin(val_patients['base_patient_id'])]
    test_files = files_df[files_df['base_patient_id'].isin(test_patients['base_patient_id'])]
    
    # Copy files to appropriate directories
    splits_data = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    copy_count = 0
    for split_name, split_df in splits_data.items():
        unique_base_patients = split_df['base_patient_id'].nunique()
        unique_full_patients = split_df['patient_id'].nunique()
        total_files = len(split_df)
        print(f"\n{split_name.capitalize()} set: {unique_base_patients} base patients, {unique_full_patients} full patient IDs, {total_files} files")
        
        label_counts = split_df['label'].value_counts().sort_index()
        for label in label_counts.index:
            base_patients_with_label = split_df[split_df['label'] == label]['base_patient_id'].nunique()
            files_with_label = label_counts[label]
            print(f"  Label {label}: {base_patients_with_label} base patients, {files_with_label} files")
        
        for _, row in split_df.iterrows():
            source_path = row['file_path']
            target_dir = os.path.join(target_dataset_folder, split_name, str(row['label']))
            target_path = os.path.join(target_dir, row['filename'])
            
            # Copy file
            shutil.copy2(source_path, target_path)
            copy_count += 1
    
    print(f"\nSuccessfully copied {copy_count} files to {target_dataset_folder}")
    print(f"Dataset structure created:")
    train_base_patients_0 = train_files[train_files['label'] == 0]['base_patient_id'].nunique()
    train_files_0 = len(train_files[train_files['label'] == 0])
    train_base_patients_1 = train_files[train_files['label'] == 1]['base_patient_id'].nunique()
    train_files_1 = len(train_files[train_files['label'] == 1])
    
    val_base_patients_0 = val_files[val_files['label'] == 0]['base_patient_id'].nunique()
    val_files_0 = len(val_files[val_files['label'] == 0])
    val_base_patients_1 = val_files[val_files['label'] == 1]['base_patient_id'].nunique()
    val_files_1 = len(val_files[val_files['label'] == 1])
    
    test_base_patients_0 = test_files[test_files['label'] == 0]['base_patient_id'].nunique()
    test_files_0 = len(test_files[test_files['label'] == 0])
    test_base_patients_1 = test_files[test_files['label'] == 1]['base_patient_id'].nunique()
    test_files_1 = len(test_files[test_files['label'] == 1])
    
    print(f"├── train/")
    print(f"│   ├── 0/ ({train_base_patients_0} base patients, {train_files_0} files)")
    print(f"│   └── 1/ ({train_base_patients_1} base patients, {train_files_1} files)")
    print(f"├── val/")
    print(f"│   ├── 0/ ({val_base_patients_0} base patients, {val_files_0} files)")
    print(f"│   └── 1/ ({val_base_patients_1} base patients, {val_files_1} files)")
    print(f"└── test/")
    print(f"    ├── 0/ ({test_base_patients_0} base patients, {test_files_0} files)")
    print(f"    └── 1/ ({test_base_patients_1} base patients, {test_files_1} files)")

# Example usage
if __name__ == "__main__":
    # Configure these paths according to your setup
    SOURCE_FOLDER = r"D:\Kananat\Data\5_adjustedBG"  # Folder containing 001, 002, etc. files
    CSV_FILE = r"D:\Kananat\Data\Classification_1.csv"       # Your CSV file with labels
    TARGET_DATASET = r"D:\Kananat\Data\training_dataset_3D\training_dataset_erosion"   # Target folder for train/val/test structure
    
    # Optional: specify file extension (e.g., '.jpg', '.png', '.npy')
    # If None, the script will try to auto-detect
    FILE_EXTENSION = ".nii.gz"  # or ".jpg", ".png", etc.
    
    # Column names in your CSV file [patient_id_column, label_column]
    COL_NAME = ['ID', 'c_erosion']  # Modify this for different CSV formats
    
    try:
        organize_dataset(
            source_folder=SOURCE_FOLDER,
            csv_file=CSV_FILE,
            target_dataset_folder=TARGET_DATASET,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            file_extension=FILE_EXTENSION,
            col_name=COL_NAME
        )
    except Exception as e:
        print(f"Error: {e}")