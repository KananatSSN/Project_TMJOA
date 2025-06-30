# exec(open(r"C:\Users\acer\Desktop\Project_TMJOA\Preprocessing\preprocessing.py").read())

import time
import slicer
import vtk
import numpy as np
import scipy.ndimage as ndi
from scipy import ndimage
import os
import nibabel as nib
from skimage import morphology
from skimage.filters import threshold_otsu
from DentalSegmentatorLib import SegmentationWidget, ExportFormat
import networkx as nx
from DICOMLib import DICOMUtils
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

def convert_dcm_to_nii(input_path, output_folder):
    """
    Load a volume from a file and return the node.
    """
    # Load the volume using Slicer's utility function
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} does not exist.")
    
    if os.path.isdir(input_path):
        dicomDataDir = os.path.join(input_path,r"_Z")
        loadedNodeIDs = []
        with DICOMUtils.TemporaryDICOMDatabase() as db:
            DICOMUtils.importDicom(dicomDataDir, db)
            patientUIDs = db.patients()
            for patientUID in patientUIDs:
                loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
        node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        slicer.app.processEvents()

        # Save node to .nii.gz
        output_name = construct_name(input_path)
        output_name = f"{output_name}.nii.gz"
        output_path = os.path.join(output_folder,output_name)
        slicer.util.saveNode(node, output_path)
        slicer.app.processEvents()
    
    slicer.mrmlScene.Clear(0)
    return output_path

def segmentation(filepath, outputfolder):

    file_name = os.path.basename(filepath)
    file_name_without_extension = file_name.split('.')[0]

    # print(f"Processing {file_name}")
    
    widget = SegmentationWidget()
    selectedFormats = ExportFormat.NIFTI 
    
    node = slicer.util.loadVolume(filepath)

    widget.inputSelector.setCurrentNode(node)
    widget.applyButton.clicked()
    widget.logic.waitForSegmentationFinished()
    slicer.app.processEvents()
    
    segmentationNode = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))[0]
    segmentation = segmentationNode.GetSegmentation()

    label = f"{file_name_without_extension}_segmentation"
    segmentId = "Segment_2" # Segment_[1-5] = ["Maxilla & Upper Skull", "Mandible", "Upper Teeth", "Lower Teeth", "Mandibular canal"]

    segment = segmentation.GetSegment(segmentId)
    if segment is None:
        print(f"Error: There is no TMJ in {file_name}")
        return
    
    segment.SetName(label)
    singleSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    singleSegmentationNode.SetName(label)
    singleSegmentationNode.GetSegmentation().AddSegment(segment)

    widget.exportSegmentation(singleSegmentationNode, outputfolder, selectedFormats)

    slicer.mrmlScene.Clear(0)
    # print(f"Finish processing {file_name}")
    return os.path.join(outputfolder, f"{file_name_without_extension}_segmentation.nii.gz")

def expand_and_fill_holes_in_segmentation(input_file, output_folder, margin_pixels=5, fill_holes=True):
    """
    Create a cover (hull) of a binary 3D volume with specified margin and fill interior holes.
    
    Parameters:
    -----------
    input_file : str
        Path to input .nii.gz file containing binary 3D volume
    output_file : str
        Path to save the resulting covered volume
    margin_pixels : int, optional
        Number of pixels for the margin (default: 5)
    fill_holes : bool, optional
        Whether to fill holes inside the object (default: True)
    """
    # Load the binary 3D volume
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Make sure the data is binary
    binary_data = (data > 0).astype(np.int16)
    
    # Keep a copy of the original binary data for statistics
    original_binary = binary_data.copy()
    
    # Fill holes if requested
    if fill_holes:
        # Fill holes in each 2D slice along all three axes for more complete filling
        filled_data = binary_data.copy()
        
        # Function to fill holes in a 3D volume by processing 2D slices
        def fill_holes_3d(volume):
            result = volume.copy()
            
            # Process along each axis
            for axis in range(3):
                # For each slice along the current axis
                slices = [result.take(i, axis=axis) for i in range(result.shape[axis])]
                
                # Fill holes in each 2D slice
                filled_slices = [ndimage.binary_fill_holes(slice) for slice in slices]
                
                # Put the filled slices back into the volume
                for i, filled_slice in enumerate(filled_slices):
                    # Create the appropriate slice object for the current axis
                    if axis == 0:
                        result[i, :, :] = filled_slice
                    elif axis == 1:
                        result[:, i, :] = filled_slice
                    else:  # axis == 2
                        result[:, :, i] = filled_slice
                        
            return result
        
        # Apply 3D hole filling
        filled_data = fill_holes_3d(binary_data)
        
        # Apply 3D binary closing to ensure complete filling of complex holes
        struct = ndimage.generate_binary_structure(3, 1)
        filled_data = ndimage.binary_closing(filled_data, structure=struct, iterations=3).astype(np.int16)
        
        # Finally, use binary_fill_holes on the entire 3D volume for any remaining holes
        filled_data = ndimage.binary_fill_holes(filled_data).astype(np.int16)
        
        # Use the filled data for further processing
        binary_data = filled_data
    
    # Create a structure element for dilation
    # Using a ball/sphere structure for 3D volumes
    struct_elem = ndimage.generate_binary_structure(3, 1)
    struct_elem = ndimage.iterate_structure(struct_elem, margin_pixels)
    
    # Dilate the binary volume to create the cover with margin
    covered_data = ndimage.binary_dilation(binary_data, structure=struct_elem).astype(np.int16)
    #print(covered_data.shape)
    # Create a new NIfTI image with the same header
    covered_img = nib.Nifti1Image(covered_data, img.affine, img.header)
    
    # Save the result
    file_name = os.path.basename(input_file)
    file_name_without_extension = file_name.rsplit('_', 1)[0]
    output_file = os.path.join(output_folder, f"{file_name_without_extension}_segmentationEdited.nii.gz")

    nib.save(covered_img, output_file)
    return output_file

def segmentation_masking(volume_path, segmentation_path, output_folder):

    # Load input volume and segmentation
    masterVolumeNode = slicer.util.loadVolume(volume_path)
    segmentationNode = slicer.util.loadSegmentation(segmentation_path)
    slicer.app.processEvents()

    # Create segment editor to get access to effects
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    # segmentEditorWidget.show() # Uncomment this line to debug thing in the program
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(segmentEditorNode)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)
    slicer.app.processEvents()

    # Set up masking parameters
    segmentEditorWidget.setActiveEffectByName("Mask volume")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("FillValue", "-4000")
    effect.setParameter("Operation", "FILL_OUTSIDE")
    maskedVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "Temporary masked volume")
    effect.self().outputVolumeSelector.setCurrentNode(maskedVolume)
    effect.self().onApply()
    slicer.app.processEvents()

    # Set up output directory
    input_file_name = os.path.basename(volume_path)
    input_file_name = os.path.splitext(input_file_name)[0]
    input_file_name = os.path.splitext(input_file_name)[0]
    output_file_name = f"{input_file_name}_masked.nii.gz"
    output_path = os.path.join(output_folder,output_file_name)

    slicer.util.saveNode(maskedVolume, output_path)
    slicer.app.processEvents()

    return output_path

def skeletonized_cropping(input_file, crop_size=50, output_folder=None, threshold=None):
    """
    Load a CBCT image, binarize it, apply 3D skeletonization, find the endpoint
    closest to (shape[0]//2, 0, 0), and crop around that point.
    
    Parameters:
    -----------
    input_file : str
        Path to the .nii.gz input file
    crop_size : int
        Size of the crop region (n) for cropping [x-n:x+n, y-n:y+n, z-n:z+n]
    output_file : str, optional
        Path to save the result. If None, will create file with '_cropped' suffix
    threshold : float, optional
        Threshold value for binarization. If None, Otsu's method will be used
    """
    # Load the NIFTI image
    # print(f"Loading {input_file}...")
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Get image properties for later saving
    affine = img.affine
    header = img.header
    
    # Determine threshold if not provided
    if threshold is None:
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(data)
        print(f"Using Otsu's threshold: {threshold}")
    
    # Convert to binary image
    # print("Converting to binary image...")
    binary = data > threshold
    
    # Apply 3D skeletonization
    # print("Applying 3D skeletonization (this may take a while)...")
    skeleton = morphology.skeletonize(binary)
    
    # Find the endpoint closest to the reference point
    reference_point = (data.shape[0]//2, 0, 0)
    # print(f"Finding endpoint closest to reference point {reference_point}...")
    endpoint = find_closest_endpoint(skeleton, reference_point)
    # print(f"Found closest endpoint at {endpoint}")
    
    # Crop the region around the endpoint
    # print(f"Cropping region of size {crop_size} around endpoint...")
    cropped_image = crop_around_point(data, endpoint, crop_size)
    cropped_image[cropped_image == 0] = -4000  # Set background to -4000
    
    # Save the full skeleton if requested
    file_name = os.path.basename(input_file)
    file_name_without_extension = file_name.rsplit('_', 1)[0]
    
    # Save the cropped region
    output_file = os.path.join(output_folder,f"{file_name_without_extension}_cropped.nii.gz")
    
    # print(f"Saving cropped region to {output_file}...")
    cropped_img = nib.Nifti1Image(cropped_image.astype(np.int16), affine, header)
    nib.save(cropped_img, output_file)
    
    # print(f"Crop coordinates: {crop_coords}")
    
    return output_file

def find_closest_endpoint(skeleton, reference_point):
    """
    Find the endpoint in the skeleton that is closest to the reference point.
    
    Parameters:
    -----------
    skeleton : numpy.ndarray
        Binary 3D array containing the skeletonized structure
    reference_point : tuple
        (x, y, z) coordinate of the reference point
    
    Returns:
    --------
    closest_endpoint : tuple
        (x, y, z) coordinate of the closest endpoint
    """
    # Create a graph from the skeleton
    G = nx.Graph()
    
    # Get coordinates of skeleton voxels
    points = np.transpose(np.where(skeleton))
    
    # Map each point to a unique node ID
    point_to_node = {}
    for i, point in enumerate(points):
        point_tuple = tuple(point)
        point_to_node[point_tuple] = i
        G.add_node(i, pos=point_tuple)
    
    # Add edges between neighboring voxels
    for point_tuple, node_id in point_to_node.items():
        x, y, z = point_tuple
        # Check 26-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    neighbor = (x + dx, y + dy, z + dz)
                    if neighbor in point_to_node:
                        G.add_edge(node_id, point_to_node[neighbor])
    
    # Find endpoints (nodes with only one connection)
    endpoints = [n for n, d in G.degree() if d == 1]
    
    if not endpoints:
        print("Warning: No endpoints found in the skeleton. Using the centroid instead.")
        # If no endpoints, use the centroid of the skeleton
        coords = np.array(np.where(skeleton)).T
        centroid = tuple(np.mean(coords, axis=0).astype(int))
        return centroid
    
    # Get coordinates of endpoints
    endpoint_coords = [G.nodes[n]['pos'] for n in endpoints]
    
    # Find the endpoint closest to the reference point
    closest_endpoint = min(endpoint_coords, 
                          key=lambda p: np.sqrt((p[0]-reference_point[0])**2 + 
                                              (p[1]-reference_point[1])**2 + 
                                              (p[2]-reference_point[2])**2))
    
    return closest_endpoint

def crop_around_point(image, point, crop_size):

    """
    Crop a region of specified size around a point.
    
    Parameters:
    -----------
    image : numpy.ndarray
        3D array to crop from
    point : tuple
        (x, y, z) coordinate of the center point
    crop_size : int
        Size of the crop region (n) for cropping [x-n:x+n, y-n:y+n, z-n:z+n]
    
    Returns:
    --------
    cropped : numpy.ndarray
        Cropped region
    crop_coords : tuple
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) coordinates of the crop
    """
    x, y, z = point
    
    # Calculate crop boundaries
    x_min = max(0, x - crop_size)
    x_max = min(image.shape[0], x + crop_size + 1)
    y_min = max(0, y - crop_size)
    y_max = min(image.shape[1], y + crop_size + 1)
    z_min = max(0, z - crop_size)
    z_max = min(image.shape[2], z + crop_size + 1)
    
    # Crop the image
    cropped = image[x_min:x_max, y_min:y_max, z_min:z_max]
    
    return cropped

def preprocessing_single(input_file, result_folder, crop_size=112, threshold=-3999):

    step_folder = ['Nii','Segmentation','Edited_Segmentation', 'Masked', 'Cropped']
    for folder in step_folder:
        os.makedirs(os.path.join(result_folder, folder), exist_ok=True)
    
    file_name = os.path.basename(input_file)
    file_name_without_extension = file_name.split('.')[0]

    total_steps = 4

    # Step 0: Convert DICOM to NIfTI if necessary
    if os.path.isdir(input_file):
        # Convert DICOM to NIfTI
        nii_folder = os.path.join(result_folder, 'Nii')
        nii_file = convert_dcm_to_nii(input_file, nii_folder)
        input_file = nii_file

    # Step 1: Segmentation
    step = 1
    bar= '▓' * step + '░' * (total_steps - step)
    print(f"Processing {file_name_without_extension} [{bar}]", end='\r')
    slicer.mrmlScene.Clear(0)
    segmentation_folder = os.path.join(result_folder, 'Segmentation')
    segmented_file = segmentation(input_file, segmentation_folder)

    # # Step 2: Augment segmentation
    step = 2
    bar= '▓' * step + '░' * (total_steps - step)
    print(f"Processing {file_name_without_extension} [{bar}]", end='\r')
    slicer.mrmlScene.Clear(0)
    edited_folder = os.path.join(result_folder, 'Edited_Segmentation')
    editedSegment_file =expand_and_fill_holes_in_segmentation(segmented_file, edited_folder, margin_pixels=5, fill_holes=True)

    # Step 3: Masking
    step = 3
    bar= '▓' * step + '░' * (total_steps - step)
    print(f"Processing {file_name_without_extension} [{bar}]", end='\r')
    slicer.mrmlScene.Clear(0)
    masked_folder = os.path.join(result_folder, 'Masked')
    masked_file = segmentation_masking(input_file, editedSegment_file, masked_folder)

    # Step 4: Skeletonization and cropping
    step = 4
    bar= '▓' * step + '░' * (total_steps - step)
    print(f"Processing {file_name_without_extension} [{bar}]", end='\r')
    slicer.mrmlScene.Clear(0)
    cropped_folder = os.path.join(result_folder, 'Cropped')
    skeletonized_cropping(masked_file, crop_size=crop_size, output_folder=cropped_folder, threshold=threshold)

    print("\n")

def batch_preprocessing(input_folder, result_folder, start_from=0, crop_size=112, threshold=-3999):

    files = sorted(os.listdir(input_folder))
    # files_count = len([filename for filename in files if filename.endswith('.nrrd')])
    # print(f"Found {files_count} .nrrd files in {input_folder}")

    progress_count = start_from
    for filename in files[start_from:]:
        if filename:
            input_file = os.path.join(input_folder, filename)
            # print(f"[Processing {progress_count} out of {files_count}]") # (Processesing)
            progress_count += 1
            preprocessing_single(input_file, result_folder, crop_size=crop_size, threshold=threshold)

resume_from = 0
input_folder = r"C:\Users\acer\Desktop\Back up\raw_Data_and_extra\raw_Data_and_extra\More_data"
result_folder = r"C:\Users\acer\Desktop\Back up\raw_Data_and_extra\raw_Data_and_extra\Preprocessed_MoreData"

batch_preprocessing(input_folder, result_folder, start_from = resume_from, crop_size=112, threshold=-3999)
# preprocessing_single(input_folder, result_folder, crop_size=112, threshold=-3999)