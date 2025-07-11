import nibabel as nib
import numpy as np
from skimage import morphology
from scipy import ndimage
import networkx as nx
import os

def process_cbct_image(input_file, crop_size=50, output_file=None, threshold=None):
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
    print(f"Loading {input_file}...")
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
    print(f"Finding endpoint closest to reference point {reference_point}...")
    endpoint = find_closest_endpoint(skeleton, reference_point)
    print(f"Found closest endpoint at {endpoint}")
    
    # Crop the region around the endpoint
    # print(f"Cropping region of size {crop_size} around endpoint...")
    cropped_image, crop_coords = crop_around_point(data, endpoint, crop_size)
    
    # Save the full skeleton if requested
    base, ext = os.path.splitext(input_file)
    if ext == '.gz':
        base, _ = os.path.splitext(base)
    
    # Save the cropped region
    if output_file is None:
        output_file = f"{base}_cropped.nii.gz"
    
    # print(f"Saving cropped region to {output_file}...")
    cropped_img = nib.Nifti1Image(cropped_image.astype(np.int16), affine, header)
    nib.save(cropped_img, output_file)
    
    # print(f"Crop coordinates: {crop_coords}")
    
    return skeleton, cropped_image, endpoint, crop_coords

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

if __name__ == "__main__":

    input_folder = r"D:\Kananat\Data\3_Registed"
    output_folder = r"D:\Kananat\Data\4_Cropped"

    nii_count = len([filename for filename in os.listdir(input_folder) if filename.endswith('.nii.gz')])
    print(f"There are {nii_count} .nii.gz files in the {input_folder}")

    progress_count = 0

    files = sorted(os.listdir(input_folder))

    for filename in files :
        if filename.endswith('.nii.gz'):
            progress_count += 1
            print(f"[Processing {progress_count} out of {nii_count}]")

            input_path = os.path.join(input_folder, filename)

            output_filename = filename.replace('_registered.nii.gz', '_cropped.nii.gz')
            output_path = os.path.join(output_folder, output_filename)

            if os.path.exists(input_path):
                # Update crop_original_image function to use the specified crop size
                process_cbct_image(input_file = input_path, crop_size=127, output_file=output_path, threshold=-3900)
                print(f"Processed {filename} and saved to {output_path}")

    print("Done")