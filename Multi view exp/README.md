# 3D Medical Image Rendering and Analysis Tool

A comprehensive Python script for 3D visualization and analysis of medical imaging data (NIfTI format), specifically designed for bone thickness analysis and temporomandibular joint (TMJ) osteoarthritis research.

## Features

- **Multiple rendering modes**: Surface, volume, slice views, and advanced depth-based analysis
- **Bidirectional depth sampling**: Analyze bone thickness by sampling in both directions from surface points
- **Interactive 3D visualization**: Rotate, zoom, and pan with real-time feedback
- **Advanced outlier handling**: Boundary clamping for cleaner visualizations
- **Statistical analysis**: Automatic histogram generation and distribution analysis
- **Clinical color mapping**: Red-to-green colorscale for intuitive bone thinning identification

## Installation Requirements

```bash
pip install numpy nibabel matplotlib plotly scikit-image scipy
```

## Basic Usage

```bash
python 3d_rendering.py "filename.nii.gz" --render_type [TYPE] [OPTIONS]
```

## Rendering Types

### 1. Binary Surface Rendering (Recommended for Bone Analysis)
```bash
python 3d_rendering.py "data.nii.gz" --render_type binary --depth_samples 15
```
- Creates clean 3D surface with optional depth-based coloring
- Best for bone thickness analysis and thinning detection

### 2. Volume Rendering
```bash
python 3d_rendering.py "data.nii.gz" --render_type volume --opacity 0.2
```
- Translucent 3D volume visualization
- Good for viewing internal structures

### 3. Surface Rendering
```bash
python 3d_rendering.py "data.nii.gz" --render_type surface
```
- Basic isosurface with intensity-based coloring
- Standard 3D surface visualization

### 4. Multi-View Slices
```bash
python 3d_rendering.py "data.nii.gz" --render_type slices
```
- Shows axial, sagittal, and coronal views simultaneously
- Good for orientation and slice-by-slice analysis

### 5. Depth-Averaged Surface
```bash
python 3d_rendering.py "data.nii.gz" --render_type depth --depth_samples 10
```
- Single viewpoint with depth-averaged surface coloring
- Colors based on sub-surface density

### 6. Multi-Viewpoint Depth Analysis
```bash
python 3d_rendering.py "data.nii.gz" --render_type multi_depth --depth_samples 10
```
- Four different camera angles showing depth-averaged colors
- Demonstrates how surface appearance changes with viewpoint

## Key Parameters

### Core Parameters
- `--depth_samples N`: Number of voxels to sample in each direction (default: 5)
- `--opacity 0.0-1.0`: Surface transparency (1.0 = solid, 0.0 = transparent)
- `--threshold N`: Percentile threshold for data filtering (default: 95)

### Visualization Options
- `--show_lines`: Display red orthogonal sampling lines
- `--line_count N`: Number of sampling lines to show (default: 50)
- `--remove_outliers`: Enable boundary clamping outlier removal

### Advanced Options
- `--downsample N`: Downsampling factor for performance (default: 2)

## How the Script Works

### 1. Data Loading and Preprocessing

**Background Detection**:
- Automatically detects background values (typically -250 for CBCT data)
- Applies exact threshold filtering to exclude background voxels
- Recenters data so minimum value becomes 0

**Data Recentering**:
```python
recentered_data = original_data - minimum_value
# Background (-250) becomes 0, anatomy becomes positive values
```

### 2. Surface Extraction

**Marching Cubes Algorithm**:
- Generates 3D surface mesh from volume data
- Uses adaptive iso-value selection:
  - Binary data: iso-value = 0.5
  - Continuous data: iso-value = 1% of data range (for complete outer surface)
- Calculates surface normals for each vertex

### 3. Bidirectional Depth Sampling

**The Core Innovation**:
For each surface vertex, the script samples voxels along the surface normal in both directions:

```
Outward ← Surface → Inward
   ↓        ↓        ↓
[n voxels] [1] [n voxels] = 2n+1 total samples
```

**Sampling Process**:
1. **Surface Normal Calculation**: Each surface vertex has an orthogonal direction vector
2. **Outward Sampling**: Sample n voxels going away from the volume
3. **Surface Vertex**: Include the surface point itself
4. **Inward Sampling**: Sample n voxels going into the volume
5. **Average Calculation**: Compute mean intensity of all 2n+1 samples

**Example with depth_samples=15**:
- Samples 15 voxels outward + 1 surface + 15 voxels inward = 31 total samples
- Each surface point color represents average density in a 31-voxel line orthogonal to surface

### 4. Color Mapping

**Clinical Color Scale**:
- **Red**: Low average values = Thin bone regions
- **Yellow**: Medium values = Moderate bone thickness  
- **Green**: High average values = Thick bone regions

**Benefits**:
- Immediate visual identification of problem areas (red)
- Intuitive interpretation (green = healthy)
- Gradual transitions show thickness variations

### 5. Outlier Removal (Optional)

**Boundary Clamping Method**:
- Identifies 5th and 95th percentiles as boundaries (keeps middle 90%)
- Low outliers (< 5th percentile) → clamped to 5th percentile value
- High outliers (> 95th percentile) → clamped to 95th percentile value
- Maintains data distribution while removing extreme noise

### 6. Statistical Analysis

**Automatic Distribution Analysis**:
- Generates histograms before and after processing
- Provides comprehensive statistics:
  - Mean, median, standard deviation
  - Percentile analysis (25th, 75th, 90th, 95th)
  - Count of non-zero values
  - Before/after comparison when outlier removal is used

## Clinical Applications

### Bone Thinning Detection
```bash
python 3d_rendering.py "tmj_scan.nii.gz" --render_type binary --depth_samples 15 --remove_outliers
```
- Optimized for detecting areas where bone is thinning
- Red areas indicate potential problem zones
- 15-voxel sampling provides ~15mm depth analysis (clinically relevant)

### Bone Quality Assessment
```bash
python 3d_rendering.py "tmj_scan.nii.gz" --render_type binary --depth_samples 20
```
- Deeper sampling for comprehensive bone quality analysis
- Useful for research applications requiring detailed density information

### Visualization with Sampling Lines
```bash
python 3d_rendering.py "tmj_scan.nii.gz" --render_type binary --depth_samples 15 --show_lines --line_count 30 --opacity 0.3
```
- Shows exactly where and how deep the sampling occurs
- Educational tool for understanding the analysis method
- Transparency allows viewing internal sampling paths

## Data Format Requirements

**Input**: NIfTI files (.nii.gz or .nii)
- Medical imaging standard format
- Supports both binary masks and grayscale intensity data
- Automatically handles different intensity ranges and background values

**Coordinate System**: 
- Uses voxel coordinates (not world coordinates)
- Results displayed in voxel units
- Suitable for relative analysis and comparison

## Performance Considerations

**Memory Usage**:
- Large volumes are downsampled by default (factor of 2)
- Bidirectional sampling is computationally intensive
- Progress indicators for long calculations

**Optimization Tips**:
- Use `--downsample 4` for very large datasets
- Reduce `--depth_samples` for faster processing
- Binary rendering is fastest, volume rendering is slowest

## Troubleshooting

**Common Issues**:

1. **Empty visualization**: Check if threshold is too high
   ```bash
   # Try lower threshold
   python 3d_rendering.py "file.nii.gz" --render_type binary --threshold 50
   ```

2. **Too noisy**: Enable outlier removal
   ```bash
   python 3d_rendering.py "file.nii.gz" --render_type binary --remove_outliers
   ```

3. **Performance issues**: Increase downsampling
   ```bash
   python 3d_rendering.py "file.nii.gz" --render_type binary --downsample 4
   ```

## Output

**Interactive 3D Visualization**:
- Opens in web browser using Plotly
- Full 3D interaction: rotate, zoom, pan
- No shadows (flat lighting for true color representation)
- Real-time navigation controls

**Statistical Reports**:
- Console output with detailed statistics
- Histogram plots showing data distribution
- Before/after analysis when using outlier removal

## Example Workflows

### Basic Bone Analysis
```bash
python 3d_rendering.py "patient_001.nii.gz" --render_type binary --depth_samples 15 --opacity 1.0
```

### Research Analysis with Full Documentation
```bash
python 3d_rendering.py "research_sample.nii.gz" --render_type binary --depth_samples 20 --remove_outliers --show_lines --line_count 50 --opacity 0.4
```

### Quick Volume Overview
```bash
python 3d_rendering.py "overview.nii.gz" --render_type volume --opacity 0.1 --threshold 70
```

## Technical Details

**Dependencies**:
- **NumPy**: Numerical computations and array operations
- **NiBabel**: NIfTI file loading and processing
- **Matplotlib**: Statistical plotting and histograms
- **Plotly**: Interactive 3D visualization
- **Scikit-image**: Marching cubes surface extraction
- **SciPy**: Image processing utilities

**Algorithm Complexity**:
- Surface extraction: O(n³) where n is volume dimension
- Depth sampling: O(v×d) where v is vertex count, d is depth samples
- For typical TMJ data: ~30 seconds for full analysis

**Data Flow**:
1. Load NIfTI → 2. Background filtering → 3. Data recentering → 4. Surface extraction → 5. Normal calculation → 6. Bidirectional sampling → 7. Statistical analysis → 8. Color mapping → 9. 3D rendering

This tool provides a comprehensive solution for medical image analysis with specific focus on bone thickness assessment and clinical visualization needs.