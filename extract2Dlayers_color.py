# 3D NIfTI to 2D PNG Layer Extractor - Top N Informative Layers
# Extracts the N most informative layers with minimum spacing between them

import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
import logging

# Additional imports for GMM processing
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set up logging for Jupyter
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================

# Dataset paths
INPUT_DATASET_DIR = r"D:\Kananat\Data\training_dataset_3D\training_dataset_subCyst"  # Update this path
OUTPUT_DATASET_DIR = r"D:\Kananat\Data\training_dataset_2D\training_dataset_subCyst"  # Update this path

# Processing parameters
N_LAYERS = 20          # Number of most informative layers to extract
MIN_SPACING = 5       # Minimum spacing between selected layers (k parameter)
BACKGROUND_VALUE = -250   # Background value to exclude

# GMM processing parameters
USE_GMM_PROCESSING = True  # Set to False to use simple normalization
DEBUG_FOLDER_PATH = None   # Set to a path like 'debug_output/' for GMM debugging, or None to disable

print("Configuration:")
print(f"Input dataset: {INPUT_DATASET_DIR}")
print(f"Output dataset: {OUTPUT_DATASET_DIR}")
print(f"Number of layers to extract: {N_LAYERS}")
print(f"Minimum spacing between layers: {MIN_SPACING}")
print(f"Background value: {BACKGROUND_VALUE}")
print(f"Use GMM processing: {USE_GMM_PROCESSING}")
print(f"Debug folder: {DEBUG_FOLDER_PATH}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Your comprehensive GMM-based layer extraction function
def extract_cbct_layer_with_gmm(nii_file_path, output_image_path, layer_number, debug_folder_path, background_value=-250):
    """
    Extract a 2D layer from 3D CBCT data with GMM-based tissue classification.
    GMM is fitted on the entire 3D volume for robust tissue classification.
    
    Parameters:
    -----------
    nii_file_path : str
        Path to the .nii.gz file
    output_image_path : str
        Path to save the output RGB image (e.g., 'output.png')
    layer_number : int
        Layer index to extract (0-based indexing)
    debug_folder_path : str
        Path to folder where debug visualizations will be saved
    background_value : float
        Background voxel value to exclude from calculations (default: -250)
    
    Returns:
    --------
    rgb_image : PIL.Image
        RGB image where:
        - Red channel: Soft tissue probability (0-255)
        - Green channel: Original voxel values (0-255)
        - Blue channel: Bone probability (0-255)
    """
    
    # Create debug folder if it doesn't exist
    if debug_folder_path:
        os.makedirs(debug_folder_path, exist_ok=True)
    
    # Load the NIfTI file
    # print(f"    Processing NIfTI file: {nii_file_path}")
    nii_img = nib.load(nii_file_path)
    volume_data = nii_img.get_fdata()
    
    # print(f"    Volume shape: {volume_data.shape}")
    # print(f"    Volume value range: [{volume_data.min():.1f}, {volume_data.max():.1f}]")
    
    # Validate layer number
    if layer_number >= volume_data.shape[0]:  # Changed from shape[2] to shape[0] for axis-0 extraction
        raise ValueError(f"Layer {layer_number} exceeds volume depth {volume_data.shape[0]}")
    
    # Remove background values from ENTIRE 3D volume for GMM fitting
    # print(f"    Preparing 3D volume data for GMM fitting...")
    volume_non_background_mask = volume_data > background_value
    volume_non_background_values = volume_data[volume_non_background_mask]
    
    # print(f"    Total voxels in volume: {volume_data.size}")
    # print(f"    Non-background voxels in volume: {len(volume_non_background_values)}")
    # print(f"    Volume non-background range: [{volume_non_background_values.min():.1f}, {volume_non_background_values.max():.1f}]")
    
    if len(volume_non_background_values) < 10:
        raise ValueError("Insufficient non-background voxels in entire volume for GMM fitting")
    
    # For very large volumes, use a random sample for GMM fitting to speed up computation
    max_samples_for_gmm = 1000000  # 1M samples should be sufficient
    if len(volume_non_background_values) > max_samples_for_gmm:
        # print(f"    Volume has {len(volume_non_background_values)} non-background voxels.")
        # print(f"    Using random sample of {max_samples_for_gmm} voxels for GMM fitting...")
        np.random.seed(42)  # For reproducible results
        sample_indices = np.random.choice(len(volume_non_background_values), 
                                        size=max_samples_for_gmm, replace=False)
        gmm_fitting_data = volume_non_background_values[sample_indices]
    else:
        gmm_fitting_data = volume_non_background_values
    
    # print(f"    Using {len(gmm_fitting_data)} voxels for GMM fitting")
    
    # Extract the specified layer for final processing (along axis 0)
    layer_2d = volume_data[layer_number, :, :]
    layer_non_background_mask = layer_2d > background_value
    # print(f"    Extracted layer {layer_number} with shape: {layer_2d.shape}")
    # print(f"    Non-background voxels in layer: {np.sum(layer_non_background_mask)}")
    
    # Fit Gaussian Mixture Model with 2 components on 3D volume data
    # print(f"    Fitting Gaussian Mixture Model on entire 3D volume...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(gmm_fitting_data.reshape(-1, 1))
    
    # Get GMM parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_
    
    # print(f"    GMM Component 1: mean={means[0]:.1f}, std={stds[0]:.1f}, weight={weights[0]:.3f}")
    # print(f"    GMM Component 2: mean={means[1]:.1f}, std={stds[1]:.1f}, weight={weights[1]:.3f}")
    
    # Determine which component is bone (higher mean) and which is soft tissue
    if means[0] > means[1]:
        bone_idx, soft_tissue_idx = 0, 1
    else:
        bone_idx, soft_tissue_idx = 1, 0
    
    # print(f"    Bone component (higher intensity): Component {bone_idx}")
    # print(f"    Soft tissue component (lower intensity): Component {soft_tissue_idx}")
    
    # Apply the trained GMM to the selected layer
    # print(f"    Applying GMM to layer {layer_number}...")
    # Calculate probabilities for all pixels in the layer
    # For background pixels, set probabilities to 0
    bone_prob = np.zeros_like(layer_2d)
    soft_tissue_prob = np.zeros_like(layer_2d)
    
    # Only calculate probabilities for non-background pixels in the layer
    if np.sum(layer_non_background_mask) > 0:
        # Get probability predictions for the entire layer
        all_probs = gmm.predict_proba(layer_2d.reshape(-1, 1))
        all_probs_reshaped = all_probs.reshape(layer_2d.shape[0], layer_2d.shape[1], 2)
        
        # Extract bone and soft tissue probabilities
        bone_prob = all_probs_reshaped[:, :, bone_idx]
        soft_tissue_prob = all_probs_reshaped[:, :, soft_tissue_idx]
        
        # Set background pixels to 0 probability
        bone_prob[~layer_non_background_mask] = 0
        soft_tissue_prob[~layer_non_background_mask] = 0
    
    # Rescale original voxel values to [0, 255] using the VOLUME range for consistency
    original_scaled = np.zeros_like(layer_2d)
    if len(volume_non_background_values) > 0:
        vol_min_val, vol_max_val = volume_non_background_values.min(), volume_non_background_values.max()
        original_scaled[layer_non_background_mask] = 255 * (layer_2d[layer_non_background_mask] - vol_min_val) / (vol_max_val - vol_min_val)
    
    # Rescale probabilities to [0, 255]
    bone_prob_scaled = (bone_prob * 255).astype(np.uint8)
    soft_tissue_prob_scaled = (soft_tissue_prob * 255).astype(np.uint8)
    original_scaled = original_scaled.astype(np.uint8)
    
    # Create RGB image
    # Red: Soft tissue probability
    # Green: Original voxel values  
    # Blue: Bone probability
    rgb_array = np.stack([soft_tissue_prob_scaled, original_scaled, bone_prob_scaled], axis=2)
    rgb_image = Image.fromarray(rgb_array, 'RGB')
    
    # Save the main RGB output image
    # print(f"    Saving RGB image to: {output_image_path}")
    rgb_image.save(output_image_path)
    
    # Save debug visualizations only if debug folder is provided

    patient_id = os.path.basename(nii_file_path)  # Use the file name as patient ID
    patient_id = patient_id.split('_')[0]  # Remove file extension

    if debug_folder_path:
        debug_plot_path = os.path.join(debug_folder_path, f'{patient_id}_gmm_analysis.png')

        if debug_folder_path and not os.path.exists(debug_plot_path):
            
            try:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Row 1: Original data and histograms
                # Original layer
                im1 = axes[0, 0].imshow(layer_2d, cmap='gray')
                axes[0, 0].set_title(f'Original Layer {layer_number}')
                axes[0, 0].axis('off')
                plt.colorbar(im1, ax=axes[0, 0])
                
                # Histogram of VOLUME non-background values (used for GMM)
                axes[0, 1].hist(gmm_fitting_data, bins=50, alpha=0.7, density=True, color='gray')
                axes[0, 1].set_title('Histogram of Volume Data (GMM Training)')
                axes[0, 1].set_xlabel('Voxel Value')
                axes[0, 1].set_ylabel('Density')
                
                # GMM overlay on histogram
                x_range = np.linspace(gmm_fitting_data.min(), gmm_fitting_data.max(), 1000)
                
                # Plot individual components
                for i in range(2):
                    component_pdf = weights[i] * (1/np.sqrt(2*np.pi*gmm.covariances_[i,0,0])) * \
                                np.exp(-0.5 * ((x_range - means[i])**2) / gmm.covariances_[i,0,0])
                    label = f'{"Bone" if i == bone_idx else "Soft Tissue"}'
                    color = 'blue' if i == bone_idx else 'red'
                    axes[0, 1].plot(x_range, component_pdf, color=color, label=label, linewidth=2)
                
                # Plot total GMM
                total_pdf = np.exp(gmm.score_samples(x_range.reshape(-1, 1)))
                axes[0, 1].plot(x_range, total_pdf, 'k--', label='Total GMM', linewidth=2)
                axes[0, 1].legend()
                
                # GMM classification result for the layer
                classification = gmm.predict(layer_2d.reshape(-1, 1)).reshape(layer_2d.shape)
                classification_display = np.full_like(layer_2d, -1, dtype=int)  # -1 for background
                classification_display[layer_non_background_mask] = classification[layer_non_background_mask]
                
                im3 = axes[0, 2].imshow(classification_display, cmap='RdBu')
                axes[0, 2].set_title(f'GMM Classification of Layer {layer_number}\n(Red=Soft Tissue, Blue=Bone)')
                axes[0, 2].axis('off')
                
                # Row 2: Probability maps and final RGB
                # Bone probability
                im4 = axes[1, 0].imshow(bone_prob, cmap='Blues', vmin=0, vmax=1)
                axes[1, 0].set_title('Bone Probability')
                axes[1, 0].axis('off')
                plt.colorbar(im4, ax=axes[1, 0])
                
                # Soft tissue probability
                im5 = axes[1, 1].imshow(soft_tissue_prob, cmap='Reds', vmin=0, vmax=1)
                axes[1, 1].set_title('Soft Tissue Probability')
                axes[1, 1].axis('off')
                plt.colorbar(im5, ax=axes[1, 1])
                
                # Final RGB result
                axes[1, 2].imshow(rgb_array)
                axes[1, 2].set_title('Final RGB Image\n(R=Soft Tissue, G=Original, B=Bone)')
                axes[1, 2].axis('off')
                
                plt.tight_layout()
                
                # Save the comprehensive debug visualization
                plt.savefig(debug_plot_path, dpi=300, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
            except Exception as e:
                print(f"    Warning: Could not create debug plot: {str(e)}")

    return rgb_image

def normalize_voxel_values(data, background_value=-250):
    """
    Normalize voxel values from [-250, max_value] to [0, 255]
    
    Args:
        data: 2D or 3D numpy array
        background_value: Background value (default: -250)
    
    Returns:
        Normalized data as uint8
    """
    # Clip values to ensure minimum is background_value
    data = np.clip(data, background_value, None)
    
    # Get min and max values
    min_val = data.min()
    max_val = data.max()
    
    if max_val == min_val:
        # Handle case where all values are the same
        return np.zeros_like(data, dtype=np.uint8)
    
    # Normalize to [0, 255]
    normalized = ((data - min_val) / (max_val - min_val)) * 255
    return normalized.astype(np.uint8)

def calculate_layer_informativeness(data, background_value=-250):
    """
    Calculate informativeness score for each layer
    Uses non-background ratio as the primary metric
    
    Args:
        data: 3D numpy array
        background_value: Background value to exclude
    
    Returns:
        List of tuples: (layer_index, informativeness_score)
    """
    layer_scores = []
    total_voxels_per_layer = data.shape[1] * data.shape[2]
    
    for i in range(data.shape[0]):
        layer = data[i, :, :]
        
        # Calculate non-background ratio
        non_background_count = np.sum(layer != background_value)
        non_background_ratio = non_background_count / total_voxels_per_layer
        
        # Additional informativeness metrics can be added here:
        # - Variance of non-background voxels
        # - Edge content
        # - Texture measures
        
        # For now, using non-background ratio as informativeness score
        informativeness_score = non_background_ratio
        
        layer_scores.append((i, informativeness_score))
    
    return layer_scores

def select_top_layers_with_spacing(layer_scores, n_layers, min_spacing):
    """
    Select top N layers ensuring minimum spacing between them
    
    Args:
        layer_scores: List of tuples (layer_index, score)
        n_layers: Number of layers to select
        min_spacing: Minimum spacing between selected layers
    
    Returns:
        List of tuples: (layer_index, score) for selected layers
    """
    # Sort by score in descending order
    sorted_layers = sorted(layer_scores, key=lambda x: x[1], reverse=True)
    
    selected_layers = []
    used_indices = set()
    
    # print(f"  Selecting {n_layers} layers with minimum spacing of {min_spacing}...")
    
    for layer_idx, score in sorted_layers:
        # Check if this layer conflicts with already selected layers
        conflict = False
        for selected_idx, _ in selected_layers:
            if abs(layer_idx - selected_idx) < min_spacing:
                conflict = True
                break
        
        if not conflict:
            selected_layers.append((layer_idx, score))
            used_indices.add(layer_idx)
            # print(f"    Selected layer {layer_idx} (score: {score:.4f})")
            
            if len(selected_layers) >= n_layers:
                break
    
    # Sort selected layers by index for consistent naming
    selected_layers.sort(key=lambda x: x[0])
    
    if len(selected_layers) < n_layers:
        print(f"    ⚠️ Only found {len(selected_layers)} layers that satisfy spacing constraint")
    
    return selected_layers

def extract_selected_layers(data, selected_layers):
    """
    Extract the selected layers
    
    Args:
        data: 3D numpy array
        selected_layers: List of tuples (layer_index, score)
    
    Returns:
        Dictionary of layer information for processing
    """
    layers = {}
    
    for i, (layer_idx, score) in enumerate(selected_layers):
        layer_key = f"layer_{layer_idx:03d}_rank_{i+1:02d}_score_{score:.3f}"
        layers[layer_key] = {
            'layer_index': layer_idx,
            'score': score,
            'rank': i + 1
        }
        # print(f"  Selected layer {layer_idx} (rank {i+1}, score: {score:.4f})")
    
    return layers

def save_layer_as_png(layer_data, output_path):
    """
    Save 2D layer as PNG with 3 identical channels (fallback method)
    
    Args:
        layer_data: 2D numpy array (normalized to 0-255)
        output_path: Output file path
    """
    # Create 3-channel image (RGB) with identical values
    rgb_image = np.stack([layer_data, layer_data, layer_data], axis=-1)
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(rgb_image, mode='RGB')
    pil_image.save(output_path)

def process_nifti_file(nifti_path, output_dir, n_layers, min_spacing, background_value=-250):
    """
    Process a single NIfTI file and extract top N informative layers with spacing
    
    Args:
        nifti_path: Path to .nii.gz file
        output_dir: Output directory for this file's layers
        n_layers: Number of layers to extract
        min_spacing: Minimum spacing between layers
        background_value: Background value
    """
    try:
        # print(f"\nProcessing: {nifti_path}")
        
        # Load NIfTI file
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        # print(f"  Data shape: {data.shape}")
        # print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
        
        # Calculate informativeness for all layers
        layer_scores = calculate_layer_informativeness(data, background_value)
        
        # Select top N layers with spacing constraint
        selected_layers = select_top_layers_with_spacing(layer_scores, n_layers, min_spacing)
        
        if not selected_layers:
            print(f"  ⚠️ No suitable layers found")
            return 0
        
        # Get layer information
        layers = extract_selected_layers(data, selected_layers)
        
        # Process and save each layer
        filename_base = Path(nifti_path).stem.replace('.nii', '')  # Remove .nii.gz extension
        filename_base = filename_base.split('_')[0]  # Remove any file extension
        saved_count = 0
        
        # Create debug folder for this file if enabled
        file_debug_folder = None
        if DEBUG_FOLDER_PATH:
            file_debug_folder = DEBUG_FOLDER_PATH
            os.makedirs(file_debug_folder, exist_ok=True)
        
        for layer_name, layer_info in layers.items():
            layer_idx = layer_info['layer_index']
            
            # Create output filename
            output_filename = f"{filename_base}_{layer_info['rank']}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                if USE_GMM_PROCESSING:
                    # Use your comprehensive GMM function
                    rgb_img = extract_cbct_layer_with_gmm(
                        nii_file_path=str(nifti_path),
                        output_image_path=output_path,
                        layer_number=layer_idx,
                        debug_folder_path=file_debug_folder,
                        background_value=background_value
                    )

                    bar_length = 10
                    filled_length = layer_info['rank']
                    
                    # Create the bar
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f"    \r   Processing {filename_base} |{bar}|", end="")
                else:
                    # Use simple normalization as fallback
                    layer_data = data[layer_idx, :, :]
                    normalized_layer = normalize_voxel_values(layer_data, background_value)
                    save_layer_as_png(normalized_layer, output_path)
                    # print(f"    ✅ Simple processed and saved: {output_filename}")
                
                saved_count += 1
                
            except Exception as layer_error:
                print(f"    ❌ Error processing layer {layer_idx}: {str(layer_error)}")
                continue
        
        # print(f"  ✅ Saved {saved_count}/{len(layers)} layers successfully")
        return saved_count
    
    except Exception as e:
        print(f"  ❌ Error processing {nifti_path}: {str(e)}")
        return 0

def process_dataset(input_dataset_dir, output_dataset_dir, n_layers, min_spacing, background_value=-250):
    """
    Process entire dataset
    
    Args:
        input_dataset_dir: Input dataset directory
        output_dataset_dir: Output dataset directory
        n_layers: Number of layers to extract per file
        min_spacing: Minimum spacing between layers
        background_value: Background value
    
    Returns:
        Dictionary with processing statistics
    """
    input_path = Path(input_dataset_dir)
    output_path = Path(output_dataset_dir)
    
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'total_layers_saved': 0,
        'splits': {}
    }
    
    # Iterate through train/val/test folders
    for split_dir in input_path.iterdir():
        if split_dir.is_dir() and split_dir.name in ['train', 'val', 'test']:
            print(f"\n{'='*50}")
            print(f"Processing split: {split_dir.name}")
            print(f"{'='*50}")
            
            stats['splits'][split_dir.name] = {'classes': {}}
            
            # Iterate through class folders (0, 1)
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir() and class_dir.name in ['0', '1']:
                    print(f"\n📁 Processing class: {class_dir.name}")
                    
                    # Create output directory
                    output_class_dir = output_path / split_dir.name / class_dir.name
                    output_class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process all .nii.gz files in this directory
                    nifti_files = list(class_dir.glob('*.nii.gz'))
                    print(f"Found {len(nifti_files)} NIfTI files")
                    
                    class_stats = {
                        'total_files': len(nifti_files),
                        'processed_files': 0,
                        'total_layers': 0
                    }
                    
                    for nifti_file in nifti_files:

                        stats['total_files'] += 1
                        layers_saved = process_nifti_file(nifti_file, output_class_dir, n_layers, min_spacing, background_value)

                        if layers_saved > 0:
                            stats['processed_files'] += 1
                            class_stats['processed_files'] += 1
                            stats['total_layers_saved'] += layers_saved
                            class_stats['total_layers'] += layers_saved
                    
                    stats['splits'][split_dir.name]['classes'][class_dir.name] = class_stats
                    print(f"Class {class_dir.name} summary: {class_stats['processed_files']}/{class_stats['total_files']} files processed, {class_stats['total_layers']} layers saved")
    
    return stats

# =============================================================================
# MAIN PROCESSING
# =============================================================================

# Validate input directory
if not os.path.exists(INPUT_DATASET_DIR):
    print(f"❌ Input dataset directory does not exist: {INPUT_DATASET_DIR}")
    print("Please update the INPUT_DATASET_DIR variable with the correct path")
else:
    print(f"✅ Input dataset found: {INPUT_DATASET_DIR}")
    
    # Validate parameters
    if N_LAYERS <= 0:
        print(f"❌ N_LAYERS must be positive, got: {N_LAYERS}")
    elif MIN_SPACING < 0:
        print(f"❌ MIN_SPACING must be non-negative, got: {MIN_SPACING}")
    else:
        print(f"✅ Starting processing to extract {N_LAYERS} layers with {MIN_SPACING} spacing...")
        
        # Process the dataset
        print(f"\n🚀 Starting dataset processing...")
        stats = process_dataset(INPUT_DATASET_DIR, OUTPUT_DATASET_DIR, N_LAYERS, MIN_SPACING, BACKGROUND_VALUE)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("📊 PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"Total layers extracted: {stats['total_layers_saved']}")
        print(f"Average layers per file: {stats['total_layers_saved']/max(stats['processed_files'], 1):.1f}")
        print(f"Output directory: {OUTPUT_DATASET_DIR}")
        
        for split_name, split_data in stats['splits'].items():
            print(f"\n{split_name.upper()}:")
            for class_name, class_data in split_data['classes'].items():
                avg_layers = class_data['total_layers'] / max(class_data['processed_files'], 1)
                print(f"  Class {class_name}: {class_data['processed_files']}/{class_data['total_files']} files → {class_data['total_layers']} layers (avg: {avg_layers:.1f})")
        
        print(f"\n✅ Processing completed successfully!")