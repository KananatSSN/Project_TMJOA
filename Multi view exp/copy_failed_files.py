import os
import shutil

source_dir = r'C:\Users\acer\Desktop\Project_TMJOA\Data\output\training_dataset_OA'
dest_dir = r'C:\Users\acer\Desktop\Project_TMJOA\Multi view exp\fail_files'

failed_files = [
    r'test\1\58-43911 L_adjustedBG.nii.gz',
    r'train\0\61-21564 R_adjustedBG.nii.gz',
    r'train\1\55-15078 R_adjustedBG.nii.gz',
    r'train\1\55-17177 L 2018_adjustedBG.nii.gz',
    r'train\1\55-17177 L 2020_adjustedBG.nii.gz',
    r'train\1\60-39505 L_adjustedBG.nii.gz',
    r'train\1\61-16015 R 2022_adjustedBG.nii.gz',
    r'train\1\61-34247 R_adjustedBG.nii.gz',
    r'train\1\64-6935 L_adjustedBG.nii.gz',
    r'val\1\64-7936 L_adjustedBG.nii.gz',
    r'val\1\64-7936 R_adjustedBG.nii.gz',
    r'val\1\65-700003 L_adjustedBG.nii.gz',
    r'val\1\65-700003 R_adjustedBG.nii.gz',
    r'val\1\66-20679 L_adjustedBG.nii.gz',
    r'val\1\66-21702 L_adjustedBG.nii.gz',
    r'val\1\66-4559 L_adjustedBG.nii.gz',
    r'val\1\66-4559 R_adjustedBG.nii.gz',
    r'val\1\66-59 L_adjustedBG.nii.gz',
    r'val\1\66-59 R_adjustedBG.nii.gz',
    r'val\1\66-700261 R_adjustedBG.nii.gz',
    r'val\1\66-700314 L_adjustedBG.nii.gz'
]

found = 0
not_found = 0

for file_path in failed_files:
    source_path = os.path.join(source_dir, file_path)
    if os.path.exists(source_path):
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy2(source_path, dest_path)
        print(f"Copied: {file_name}")
        found += 1
    else:
        print(f"Not found: {file_path}")
        not_found += 1

print(f"\nSummary: {found} files copied, {not_found} files not found")