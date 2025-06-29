# Complete Segmentation and Cropping Pipeline for 3D Slicer
# Run this in 3D Slicer's Python console
# exec(open(r"C:\Users\acer\Desktop\Project_TMJOA\Preprocessing\preprocessing.py").read())

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

def complete_segmentation_pipeline(input_path, outputfolder, crop_size=112, threshold=None):
    """
    Complete pipeline: Segment -> Fill holes -> Expand -> Extract -> Crop to 224x224x224
    
    Parameters:
    -----------
    input_path : str
        Path to input medical image (.nii.gz, .nrrd) or folder containing .dcm files
    outputfolder : str  
        Output directory
    crop_size : int
        Half size for cropping (final size will be 2*crop_size = 224)
    threshold : float
        Threshold for skeletonization (optional)
    """
    
    # Step 1: Load the input data (handles multiple formats)
    print("Step 1: Loading input data...")
    node, input_name = load_medical_image(input_path)
    
    if node is None:
        print(f"Error: Could not load {input_path}")
        return None
        
    print(f"Processing {input_name}")
    
    # Step 2: Run segmentation
    print("Step 2: Running segmentation...")
    widget = SegmentationWidget()
    selectedFormats = ExportFormat.NIFTI
    
    widget.inputSelector.setCurrentNode(node)
    widget.applyButton.clicked()
    widget.logic.waitForSegmentationFinished()
    slicer.app.processEvents()
    
    # Get segmentation result
    segmentationNode = list(slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode"))[0]
    segmentation = segmentationNode.GetSegmentation()
    segmentId = "Mandible"  # Mandible
    segment = segmentation.GetSegment(segmentId)
    
    if segment is None:
        print(f"Error: There is no mandible segment in {input_name}")
        cleanup_nodes([segmentationNode, node])
        return None
    
    # Convert segmentation to labelmap
    print("Step 3: Converting segmentation to labelmap...")
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
        segmentationNode, labelmapVolumeNode)
    
    # Get arrays
    original_array = slicer.util.arrayFromVolume(node)
    segmentation_array = slicer.util.arrayFromVolume(labelmapVolumeNode)
    
    # Get volume properties (robust method for different Slicer versions)
    spacing = node.GetSpacing()
    origin = node.GetOrigin()
    
    # Get directions matrix (compatible approach)
    try:
        # Try the newer method first
        directions_matrix = vtk.vtkMatrix4x4()
        node.GetIJKToRASDirectionMatrix(directions_matrix)
    except:
        try:
            # Try alternative method
            directions_matrix = vtk.vtkMatrix4x4()
            node.GetIJKToRASMatrix(directions_matrix)
        except:
            # Fallback to identity matrix
            directions_matrix = vtk.vtkMatrix4x4()
            directions_matrix.Identity()
    
    print("Step 4: Processing segmentation mask...")
    
    # Get arrays
    original_array = slicer.util.arrayFromVolume(node)
    segmentation_array = slicer.util.arrayFromVolume(labelmapVolumeNode)
    
    print(f"Original array shape: {original_array.shape}")
    print(f"Segmentation array shape: {segmentation_array.shape}")
    
    # Check if dimensions match, if not, resample segmentation to match original
    if original_array.shape != segmentation_array.shape:
        print("Dimension mismatch detected. Resampling segmentation to match original volume...")
        
        # Create a new labelmap with matching dimensions
        resampledLabelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        resampledLabelmapNode.SetName("ResampledSegmentation")
        
        # Set up resampling parameters
        parameters = {
            "inputVolume": labelmapVolumeNode.GetID(),
            "referenceVolume": node.GetID(),  # Use original volume as reference
            "outputVolume": resampledLabelmapNode.GetID(),
            "interpolationMode": "NearestNeighbor",  # Important for label maps
            "defaultValue": 0
        }
        
        # Run resampling
        resampleModule = slicer.modules.resamplescalarvectordwivolume
        cliNode = slicer.cli.run(resampleModule, None, parameters, wait_for_completion=True)
        
        # Use resampled segmentation
        segmentation_array = slicer.util.arrayFromVolume(resampledLabelmapNode)
        print(f"Resampled segmentation shape: {segmentation_array.shape}")
        
        # Clean up
        slicer.mrmlScene.RemoveNode(resampledLabelmapNode)
        slicer.mrmlScene.RemoveNode(cliNode)
    
    # Ensure arrays have exactly the same shape
    if original_array.shape != segmentation_array.shape:
        print("Error: Unable to match array dimensions. Trying manual resize...")
        # Manual resize as last resort
        from scipy.ndimage import zoom
        
        zoom_factors = [orig/seg for orig, seg in zip(original_array.shape, segmentation_array.shape)]
        segmentation_array = zoom(segmentation_array, zoom_factors, order=0)  # Nearest neighbor
        segmentation_array = segmentation_array.astype(np.uint8)
        print(f"Manually resized segmentation shape: {segmentation_array.shape}")
    
    # Step 2: Fill holes in segmentation
    binary_mask = segmentation_array > 0
    filled_mask = ndi.binary_fill_holes(binary_mask)
    print("   - Filled holes in segmentation")
    
    # Step 3: Expand border by 10 pixels (morphological dilation)
    structure = ndi.generate_binary_structure(3, 1)  # 3D connectivity
    expanded_mask = ndi.binary_dilation(filled_mask, structure=structure, iterations=10)
    print("   - Expanded borders by 10 pixels")
    
    # Step 4: Extract original image data using the mask and create new image with -2000 background
    print("Step 5: Extracting masked region...")
    
    # Final check to ensure arrays match exactly
    if original_array.shape != expanded_mask.shape:
        print(f"Final shape mismatch: original {original_array.shape} vs mask {expanded_mask.shape}")
        # Crop or pad the mask to match original exactly
        expanded_mask = match_array_shapes(expanded_mask, original_array.shape)
        print(f"Adjusted mask shape: {expanded_mask.shape}")
    
    # Create new image with -2000 background
    extracted_image = np.full_like(original_array, -2000, dtype=np.float32)
    
    # Copy original values where mask is True
    extracted_image[expanded_mask] = original_array[expanded_mask]
    
    # Create volume node for the extracted image
    extracted_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    extracted_node.SetName(f"{input_name}_extracted")
    extracted_node.SetSpacing(spacing)
    extracted_node.SetOrigin(origin)
    
    # Set directions matrix (robust method)
    try:
        extracted_node.SetIJKToRASDirectionMatrix(directions_matrix)
    except:
        try:
            extracted_node.SetIJKToRASMatrix(directions_matrix)
        except:
            # Fallback - copy geometry from original node
            extracted_node.Copy(node)
            extracted_node.SetName(f"{input_name}_extracted")
    
    slicer.util.updateVolumeFromArray(extracted_node, extracted_image)
    
    # Step 5: Apply the CBCT processing function
    print("Step 6: Applying CBCT skeletonization and cropping...")
    
    # Save temporary file for the CBCT processing function
    temp_extracted_path = os.path.join(outputfolder, f"{input_name}_temp_extracted.nii.gz")
    slicer.util.saveNode(extracted_node, temp_extracted_path)
    
    # Apply the CBCT processing
    try:
        skeleton, cropped_image, endpoint, crop_coords = process_cbct_image_slicer(
            temp_extracted_path, crop_size=crop_size, threshold=threshold
        )
        print(f"   - Found endpoint at: {endpoint}")
        print(f"   - Crop coordinates: {crop_coords}")
        
        # Ensure the cropped image is exactly 224x224x224
        final_cropped = ensure_224_size(cropped_image)
        
        # Save the final result
        output_file = os.path.join(outputfolder, f"{input_name}_cropped_224.nii.gz")
        
        # Create NIfTI image and save
        # Use identity affine for the cropped image since we're changing dimensions
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1] 
        affine[2, 2] = spacing[2]
        
        final_img = nib.Nifti1Image(final_cropped.astype(np.int16), affine)
        nib.save(final_img, output_file)
        
        print(f"Step 7: Saved final result to {output_file}")
        print(f"Final image shape: {final_cropped.shape}")
        
        # Clean up temporary file
        if os.path.exists(temp_extracted_path):
            os.remove(temp_extracted_path)
            
    except Exception as e:
        print(f"Error in CBCT processing: {e}")
        cleanup_nodes([segmentationNode, labelmapVolumeNode, extracted_node, node])
        return None
    
    # Cleanup Slicer nodes
    cleanup_nodes([segmentationNode, labelmapVolumeNode, extracted_node, node])
    
    print(f"Finished processing {input_name}")
    return output_file

def load_medical_image(input_path):
    """
    Load medical image from various formats (.nii.gz, .nrrd, or DICOM folder)
    
    Parameters:
    -----------
    input_path : str
        Path to file or folder
        
    Returns:
    --------
    node : vtkMRMLScalarVolumeNode or None
        Loaded volume node
    name : str
        Base name for output files
    """
    from DICOMLib import DICOMUtils
    
    if os.path.isfile(input_path):
        # Single file (.nii.gz, .nrrd, etc.)
        file_name = os.path.basename(input_path)
        name_without_extension = os.path.splitext(file_name)[0]
        if name_without_extension.endswith('.nii'):
            name_without_extension = os.path.splitext(name_without_extension)[0]
        
        try:
            node = slicer.util.loadVolume(input_path)
            print(f"Loaded single file: {file_name}")
            return node, name_without_extension
        except Exception as e:
            print(f"Error loading file {input_path}: {e}")
            return None, None
            
    elif os.path.isdir(input_path):
        # Directory - check if it contains DICOM files
        dcm_files = [f for f in os.listdir(input_path) if f.lower().endswith('.dcm')]
        
        if dcm_files:
            # Load DICOM folder
            try:
                loadedNodeIDs = []
                with DICOMUtils.TemporaryDICOMDatabase() as db:
                    DICOMUtils.importDicom(input_path, db)
                    patientUIDs = db.patients()
                    for patientUID in patientUIDs:
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
                
                node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                folder_name = os.path.basename(input_path.rstrip(os.sep))
                print(f"Loaded DICOM folder: {folder_name} ({len(dcm_files)} files)")
                return node, folder_name
                
            except Exception as e:
                print(f"Error loading DICOM folder {input_path}: {e}")
                return None, None
        else:
            # Check for other medical image files in the directory
            medical_extensions = ['.nii.gz', '.nrrd', '.nii', '.mha', '.mhd']
            medical_files = []
            for ext in medical_extensions:
                medical_files.extend([f for f in os.listdir(input_path) if f.lower().endswith(ext)])
            
            if medical_files:
                # Load the first medical image file found
                first_file = medical_files[0]
                file_path = os.path.join(input_path, first_file)
                try:
                    node = slicer.util.loadVolume(file_path)
                    name_without_extension = os.path.splitext(first_file)[0]
                    if name_without_extension.endswith('.nii'):
                        name_without_extension = os.path.splitext(name_without_extension)[0]
                    print(f"Loaded medical image from directory: {first_file}")
                    return node, name_without_extension
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    return None, None
            else:
                print(f"No medical image files found in directory: {input_path}")
                return None, None
    else:
        print(f"Path does not exist: {input_path}")
        return None, None

def cleanup_nodes(nodes):
    """Helper function to clean up Slicer nodes"""
    for node in nodes:
        if node is not None:
            try:
                slicer.mrmlScene.RemoveNode(node)
            except:
                pass  # Node might already be removed

def match_array_shapes(source_array, target_shape):
    """
    Resize source array to match target shape exactly by cropping or padding
    """
    current_shape = source_array.shape
    
    # If shapes already match, return as is
    if current_shape == target_shape:
        return source_array
    
    # Create output array with target shape
    output = np.zeros(target_shape, dtype=source_array.dtype)
    
    # Calculate slices for each dimension
    slices_source = []
    slices_target = []
    
    for i in range(len(target_shape)):
        if current_shape[i] >= target_shape[i]:
            # Need to crop source
            start = (current_shape[i] - target_shape[i]) // 2
            slices_source.append(slice(start, start + target_shape[i]))
            slices_target.append(slice(0, target_shape[i]))
        else:
            # Need to pad (center the source in target)
            start = (target_shape[i] - current_shape[i]) // 2
            slices_source.append(slice(0, current_shape[i]))
            slices_target.append(slice(start, start + current_shape[i]))
    
    # Copy data
    output[tuple(slices_target)] = source_array[tuple(slices_source)]
    
    return output
    """
    Detect the type of input data
    
    Returns:
    --------
    str : 'file', 'dicom_folder', 'medical_folder', or 'unknown'
    """
    if os.path.isfile(input_path):
        return 'file'
    elif os.path.isdir(input_path):
        files = os.listdir(input_path)
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        medical_extensions = ['.nii.gz', '.nrrd', '.nii', '.mha', '.mhd']
        medical_files = []
        for ext in medical_extensions:
            medical_files.extend([f for f in files if f.lower().endswith(ext)])
        
        if dcm_files:
            return 'dicom_folder'
        elif medical_files:
            return 'medical_folder'
        else:
            return 'unknown'
    else:
        return 'unknown'

def process_cbct_image_slicer(input_file, crop_size=112, threshold=None):
    """
    Modified version of the CBCT processing function for use in Slicer
    """
    print(f"Loading {input_file}...")
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Determine threshold if not provided
    if threshold is None:
        # Only calculate threshold on non-background voxels
        non_background = data[data > -1900]  # Exclude -2000 background
        if len(non_background) > 0:
            threshold = threshold_otsu(non_background)
        else:
            threshold = 0
        print(f"Using Otsu's threshold: {threshold}")
    
    # Convert to binary image (exclude background)
    binary = (data > threshold) & (data > -1900)  # Exclude -2000 background
    
    if not np.any(binary):
        print("Warning: No voxels above threshold found!")
        # Return center crop if no structure found
        center = [s//2 for s in data.shape]
        cropped_image, crop_coords = crop_around_point(data, center, crop_size)
        return None, cropped_image, center, crop_coords
    
    # Apply 3D skeletonization
    print("Applying 3D skeletonization...")
    skeleton = morphology.skeletonize(binary)
    
    # Find the endpoint closest to the reference point
    reference_point = (data.shape[0]//2, 0, 0)
    print(f"Finding endpoint closest to reference point {reference_point}...")
    endpoint = find_closest_endpoint(skeleton, reference_point)
    print(f"Found closest endpoint at {endpoint}")
    
    # Crop the region around the endpoint
    print(f"Cropping region of size {crop_size*2} around endpoint...")
    cropped_image, crop_coords = crop_around_point(data, endpoint, crop_size)
    
    return skeleton, cropped_image, endpoint, crop_coords

def find_closest_endpoint(skeleton, reference_point):
    """Find the endpoint in the skeleton closest to reference point"""
    if not np.any(skeleton):
        print("Warning: Empty skeleton, using reference point")
        return reference_point
        
    # Create a graph from the skeleton
    G = nx.Graph()
    
    # Get coordinates of skeleton voxels
    points = np.transpose(np.where(skeleton))
    
    if len(points) == 0:
        return reference_point
    
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
        print("Warning: No endpoints found, using centroid")
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
    """Crop a region around a point"""
    x, y, z = point
    
    # Calculate crop boundaries
    x_min = max(0, x - crop_size)
    x_max = min(image.shape[0], x + crop_size)
    y_min = max(0, y - crop_size)
    y_max = min(image.shape[1], y + crop_size)
    z_min = max(0, z - crop_size)
    z_max = min(image.shape[2], z + crop_size)
    
    # Crop the image
    cropped = image[x_min:x_max, y_min:y_max, z_min:z_max]
    
    crop_coords = ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    return cropped, crop_coords

def ensure_224_size(image, target_size=224):
    """
    Ensure the image is exactly 224x224x224 by padding or cropping
    """
    current_shape = image.shape
    
    # If already 224x224x224, return as is
    if all(s == target_size for s in current_shape):
        return image
    
    # Create output array filled with -2000
    output = np.full((target_size, target_size, target_size), -2000, dtype=image.dtype)
    
    # Calculate how much to take from input and where to place it
    for dim in range(3):
        if current_shape[dim] >= target_size:
            # Need to crop
            start_input = (current_shape[dim] - target_size) // 2
            end_input = start_input + target_size
            start_output = 0
            end_output = target_size
        else:
            # Need to pad
            start_input = 0
            end_input = current_shape[dim]
            start_output = (target_size - current_shape[dim]) // 2
            end_output = start_output + current_shape[dim]
        
        if dim == 0:
            input_slice = slice(start_input, end_input)
            output_slice = slice(start_output, end_output)
            x_in, x_out = input_slice, output_slice
        elif dim == 1:
            input_slice = slice(start_input, end_input)
            output_slice = slice(start_output, end_output)
            y_in, y_out = input_slice, output_slice
        else:
            input_slice = slice(start_input, end_input)
            output_slice = slice(start_output, end_output)
            z_in, z_out = input_slice, output_slice
    
    # Copy the data
    output[x_out, y_out, z_out] = image[x_in, y_in, z_in]
    
    print(f"Resized from {current_shape} to {output.shape}")
    return output

# Example usage function
def batch_process_inputs(input_paths, output_folder):
    """
    Process multiple inputs in batch - handles mixed input types
    
    Parameters:
    -----------
    input_paths : list
        List of file paths or folder paths
    output_folder : str
        Output directory
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    successful = 0
    failed = 0
    
    for input_path in input_paths:
        input_type = detect_input_type(input_path)
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        print(f"Detected type: {input_type}")
        print(f"{'='*60}")
        
        try:
            result = complete_segmentation_pipeline(input_path, output_folder)
            if result:
                print(f"✓ Successfully processed: {os.path.basename(input_path)}")
                successful += 1
            else:
                print(f"✗ Failed to process: {os.path.basename(input_path)}")
                failed += 1
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(input_path)}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"{'='*60}")

def batch_process_folder(input_folder, output_folder, recursive=False):
    """
    Process all medical images in a folder
    
    Parameters:
    -----------
    input_folder : str
        Folder containing medical images or DICOM folders
    output_folder : str
        Output directory
    recursive : bool
        Whether to search subfolders recursively
    """
    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        return
    
    input_paths = []
    
    if recursive:
        # Search recursively
        for root, dirs, files in os.walk(input_folder):
            # Check for medical image files
            medical_extensions = ['.nii.gz', '.nrrd', '.nii', '.mha', '.mhd']
            for file in files:
                if any(file.lower().endswith(ext) for ext in medical_extensions):
                    input_paths.append(os.path.join(root, file))
            
            # Check for DICOM folders
            dcm_files = [f for f in files if f.lower().endswith('.dcm')]
            if dcm_files:
                input_paths.append(root)
    else:
        # Search only in the main folder
        items = os.listdir(input_folder)
        
        for item in items:
            item_path = os.path.join(input_folder, item)
            
            if os.path.isfile(item_path):
                # Check if it's a medical image file
                medical_extensions = ['.nii.gz', '.nrrd', '.nii', '.mha', '.mhd']
                if any(item.lower().endswith(ext) for ext in medical_extensions):
                    input_paths.append(item_path)
                    
            elif os.path.isdir(item_path):
                # Check if folder contains DICOM files
                try:
                    folder_files = os.listdir(item_path)
                    dcm_files = [f for f in folder_files if f.lower().endswith('.dcm')]
                    if dcm_files:
                        input_paths.append(item_path)
                except:
                    continue
    
    if not input_paths:
        print(f"No medical images or DICOM folders found in: {input_folder}")
        return
    
    print(f"Found {len(input_paths)} inputs to process:")
    for path in input_paths:
        input_type = detect_input_type(path)
        print(f"  - {os.path.basename(path)} ({input_type})")
    
    batch_process_inputs(input_paths, output_folder)

input_path = r"C:\Users\acer\Desktop\Archieve\Open access data\Baseline\Baseline\001_left_baseline_mirrored.nrrd"
outputfolder = r"C:\Users\acer\Desktop\Archieve\Open access data\Baseline\result"
complete_segmentation_pipeline(input_path, outputfolder)

# Quick usage examples:
"""
# Process single .nii.gz file:
input_path = r"C:\path\to\your\input.nii.gz"
outputfolder = r"C:\path\to\output"
complete_segmentation_pipeline(input_path, outputfolder)

# Process single .nrrd file:
input_path = r"C:\path\to\your\input.nrrd"
outputfolder = r"C:\path\to\output"
complete_segmentation_pipeline(input_path, outputfolder)

# Process DICOM folder:
input_path = r"C:\path\to\dicom\folder"
outputfolder = r"C:\path\to\output"
complete_segmentation_pipeline(input_path, outputfolder)

# Process multiple mixed inputs:
input_paths = [
    r"C:\path\to\file1.nii.gz",
    r"C:\path\to\file2.nrrd", 
    r"C:\path\to\dicom\folder1",
    r"C:\path\to\dicom\folder2"
]
output_folder = r"C:\path\to\output"
batch_process_inputs(input_paths, output_folder)

# Process entire folder (non-recursive):
input_folder = r"C:\path\to\input\folder"
output_folder = r"C:\path\to\output\folder" 
batch_process_folder(input_folder, output_folder, recursive=False)

# Process entire folder tree (recursive):
batch_process_folder(input_folder, output_folder, recursive=True)
"""

print("Complete segmentation pipeline loaded!")
print("Use: complete_segmentation_pipeline(input_path, outputfolder)")
print("\nSupported input formats:")
print("• .nii.gz files")
print("• .nrrd files") 
print("• .nii files")
print("• .mha/.mhd files")
print("• Folders containing .dcm files")
print("• Folders containing medical image files")
print("\nThe pipeline will:")
print("1. Load input (auto-detects format)")
print("2. Segment the image (extract mandible)")
print("3. Fill holes in the segmentation")
print("4. Expand borders by 10 pixels")
print("5. Extract original image data with -2000 background")
print("6. Apply skeletonization and find optimal crop point")
print("7. Crop to 224x224x224 and save as .nii.gz")
print("\nBatch processing options:")
print("• batch_process_inputs(list_of_paths, output_folder)")
print("• batch_process_folder(input_folder, output_folder, recursive=True/False)")