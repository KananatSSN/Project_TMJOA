from pathlib import Path

def validate_data_structure(data_path):
    """Validate the expected data structure"""
    data_path = Path(data_path)
    
    required_splits = ['train', 'val', 'test']
    required_classes = ['0', '1']
    
    print(f"Validating data structure at {data_path}...")
    
    all_valid = True
    for split in required_splits:
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ Missing split folder: {split_path}")
            all_valid = False
            continue
            
        for class_label in required_classes:
            class_path = split_path / class_label
            if not class_path.exists():
                print(f"❌ Missing class folder: {class_path}")
                all_valid = False
            else:
                # Count files in this class
                files = list(class_path.glob('*'))
                valid_files = [f for f in files if any(str(f).lower().endswith(ext) 
                              for ext in ['.nii', '.nii.gz', '.npy', '.npz', '.dcm'])]
                print(f"✅ {split}/{class_label}: {len(valid_files)} valid files")
    
    if all_valid:
        print("✅ Data structure validation passed!")
    else:
        print("❌ Data structure validation failed!")
        print("\nExpected structure:")
        print("data_path/")
        print("├── train/")
        print("│   ├── 0/  (class 0 training files)")
        print("│   └── 1/  (class 1 training files)")
        print("├── val/")
        print("│   ├── 0/  (class 0 validation files)")
        print("│   └── 1/  (class 1 validation files)")
        print("└── test/")
        print("    ├── 0/  (class 0 test files)")
        print("    └── 1/  (class 1 test files)")
    
    return all_valid