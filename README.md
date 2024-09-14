# Automated detection of TMJOA with CBCT using AI/ML
## Installation
1. Request data from kananat.ssn@gmail.com
2. Download and install 3dslicer version 5.7.0 (or higher) from https://download.slicer.org/
3. Install DentalSegmentator module in 3dslicer. (https://github.com/gaudot/SlicerDentalSegmentator)
## Phase 0
1. Run dcm_to_nii.py in 3dslicer python console to convert Dicom file to NlfTl.
## Phase 1
1. Run dcm_to_nii.py in 3dslicer python console to segment the mandibles.
2. If you don't have GPU, use auto_yes.py to enable batch processing.
## Phase 1.5
1. Use fill_inside_labeled_area.ipynb to edit segmentation.
2. Run masking.py in 3dslicer python console to mask the segmentation area from original image.
3. Use squeezing.ipynb to squeeze out the slices with no information.
