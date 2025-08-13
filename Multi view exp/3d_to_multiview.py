#!/usr/bin/env python3
"""
3D to Multi-view Generator

Simplified script to convert 3D NIfTI volumes to multi-view 2D images
with predefined optimal settings for medical imaging analysis.

Based on 3d_rendering.py but streamlined for batch multi-view generation.

Usage:
    # Single file processing
    python 3d_to_multiview.py input_file.nii.gz output_folder
    python 3d_to_multiview.py input_file.nii.gz output_folder --rotation_step 45
    
    # Batch processing (preserves dataset structure)
    python 3d_to_multiview.py --batch input_dataset_root output_dataset_root
    python 3d_to_multiview.py --batch input_dataset_root output_dataset_root --rotation_step 30
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from scipy import ndimage
import argparse
import os
import math
import time
import sys
import gc
from pathlib import Path
import shutil
from typing import List, Tuple
import psutil
import warnings

class MultiViewGenerator:
    def __init__(self, nifti_path, threshold_percentile=95):
        """
        Initialize the multi-view generator
        
        Args:
            nifti_path (str): Path to the .nii.gz file
            threshold_percentile (float): Percentile for intensity thresholding (default: 95)
        """
        self.nifti_path = nifti_path
        self.threshold_percentile = threshold_percentile
        self.img_data = None
        self.img_data_thresholded = None
        self.affine = None
        self.header = None
        self._pre_calculated_surface = None
        self._nib_image = None  # Keep reference to nibabel image for proper cleanup
        
        self.load_nifti()
    
    def load_nifti(self):
        """Load NIfTI image data with memory monitoring and robust error handling"""
        try:
            # Check if file exists and is readable
            if not os.path.exists(self.nifti_path):
                raise FileNotFoundError(f"File not found: {self.nifti_path}")
            
            if not os.access(self.nifti_path, os.R_OK):
                raise PermissionError(f"File not readable: {self.nifti_path}")
            
            # Monitor memory before loading
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load with explicit error handling for different failure modes
            try:
                self._nib_image = nib.load(self.nifti_path)
            except Exception as load_error:
                raise Exception(f"Could not read file: {self.nifti_path}. Nibabel error: {str(load_error)}")
            
            try:
                # Use float32 to save memory, with explicit dtype conversion
                self.img_data = self._nib_image.get_fdata(dtype=np.float32, caching='unchanged')
                if self.img_data is None:
                    raise ValueError("Image data is None")
                    
            except Exception as data_error:
                raise Exception(f"Could not extract image data: {str(data_error)}")
            
            # Copy metadata
            try:
                self.affine = self._nib_image.affine.copy()
                self.header = self._nib_image.header.copy()
            except Exception as meta_error:
                warnings.warn(f"Could not copy metadata: {str(meta_error)}")
                self.affine = np.eye(4)
                self.header = None
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Loaded NIfTI image: {os.path.basename(self.nifti_path)}")
            print(f"Shape: {self.img_data.shape}")
            print(f"Value range: {self.img_data.min():.2f} to {self.img_data.max():.2f}")
            print(f"Memory usage: {mem_after - mem_before:.1f} MB")
            
        except Exception as e:
            self.cleanup()
            # Re-raise with more context
            error_msg = f"Error loading NIfTI file: {str(e)}"
            if "Could not read file" not in str(e):
                error_msg = f"Error loading NIfTI file: Could not read file: {self.nifti_path}"
            raise Exception(error_msg)
    
    def preprocess_data(self, downsample_factor=2, flip_upside_down=True):
        """
        Preprocess the data for 3D visualization
        
        Args:
            downsample_factor (int): Factor to downsample the data for performance
            flip_upside_down (bool): Whether to flip the volume upside down
        """
        # Flip volume upside down (default enabled for medical imaging)
        if flip_upside_down:
            print("Flipping volume upside down...")
            self.img_data = np.flip(self.img_data, axis=2)
        
        # Downsample for performance if needed
        if downsample_factor > 1:
            self.img_data = self.img_data[::downsample_factor, 
                                        ::downsample_factor, 
                                        ::downsample_factor]
            print(f"Downsampled to shape: {self.img_data.shape}")
        
        # Apply threshold to focus on relevant structures
        min_val = self.img_data.min()
        
        # If minimum value is -250, use it as exact threshold for informative voxels
        if min_val == -250:
            threshold = -250
            print(f"Using exact background threshold: {threshold:.1f}")
        elif min_val < -100:
            print(f"Detected background value: {min_val:.1f}")
            background_threshold = min_val + 50
            
            non_background_voxels = self.img_data[self.img_data > background_threshold]
            if len(non_background_voxels) > 0:
                threshold = np.percentile(non_background_voxels, self.threshold_percentile)
            else:
                threshold = background_threshold
                
            print(f"Using background-aware threshold: {threshold:.4f}")
        else:
            if np.any(self.img_data > 0):
                threshold = np.percentile(self.img_data[self.img_data > 0], self.threshold_percentile)
            else:
                threshold = np.percentile(self.img_data, self.threshold_percentile)
            print(f"Using percentile threshold: {threshold:.4f}")
        
        self.img_data_thresholded = np.where(self.img_data > threshold, self.img_data, 0)
        print(f"Non-zero voxels after thresholding: {np.count_nonzero(self.img_data_thresholded)}")
    
    def _calculate_depth_sum_colors(self, vertices, normals, depth_samples, remove_outliers=True, outlier_percent=10.0):
        """
        Calculate depth-sum intensity colors for surface vertices
        """
        print("Calculating bidirectional depth-average colors...")
        print(f"Sampling {depth_samples} voxels outward + surface + {depth_samples} voxels inward = {2*depth_samples+1} total samples")
        
        # Recenter the data so minimum value becomes 0
        min_val = self.img_data.min()
        recentered_data = self.img_data - min_val
        
        colors = np.zeros(len(vertices))
        
        for i, (vertex, normal) in enumerate(zip(vertices, normals)):
            if i % 10000 == 0:
                print(f"Processing vertex {i}/{len(vertices)}")
            
            # Normalize the surface normal
            normal = normal / np.linalg.norm(normal)
            
            # Sample points along the normal in both directions
            intensities = []
            
            # Sample outward from surface
            for d in range(-depth_samples, 0):
                sample_point = vertex + normal * d
                intensity = self._interpolate_intensity(sample_point, recentered_data)
                intensities.append(intensity)
            
            # Add the surface vertex itself
            intensity = self._interpolate_intensity(vertex, recentered_data)
            intensities.append(intensity)
            
            # Sample inward from surface
            for d in range(1, depth_samples + 1):
                sample_point = vertex + normal * d
                intensity = self._interpolate_intensity(sample_point, recentered_data)
                intensities.append(intensity)
            
            # Average the intensities
            colors[i] = np.mean(intensities) if intensities else 0
        
        print(f"Bidirectional depth-average color range: {colors.min():.2f} to {colors.max():.2f}")
        
        if remove_outliers:
            # Remove outliers by boundary clamping
            half_outlier = outlier_percent / 2.0
            lower_bound = np.percentile(colors, half_outlier)
            upper_bound = np.percentile(colors, 100.0 - half_outlier)
            
            colors_clipped = np.clip(colors, lower_bound, upper_bound)
            
            print(f"Outlier removal: clamped to [{lower_bound:.2f}, {upper_bound:.2f}]")
            return colors_clipped
        else:
            return colors
    
    def _interpolate_intensity(self, point, data):
        """Trilinear interpolation for smooth intensity sampling"""
        x, y, z = point
        
        # Check bounds
        if (x < 0 or x >= data.shape[0] - 1 or 
            y < 0 or y >= data.shape[1] - 1 or 
            z < 0 or z >= data.shape[2] - 1):
            return 0.0
        
        # Get integer coordinates and fractional parts
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = min(x0 + 1, data.shape[0] - 1), min(y0 + 1, data.shape[1] - 1), min(z0 + 1, data.shape[2] - 1)
        
        xd, yd, zd = x - x0, y - y0, z - z0
        
        # Trilinear interpolation
        c000 = data[x0, y0, z0]
        c001 = data[x0, y0, z1]
        c010 = data[x0, y1, z0]
        c011 = data[x0, y1, z1]
        c100 = data[x1, y0, z0]
        c101 = data[x1, y0, z1]
        c110 = data[x1, y1, z0]
        c111 = data[x1, y1, z1]
        
        # Interpolate in x direction
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        # Interpolate in y direction
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        # Final interpolation in z direction
        return c0 * (1 - zd) + c1 * zd
    
    def _apply_custom_color_mapping(self, colors, color_min=None, color_max=None):
        """Apply custom color range mapping to depth colors"""
        if color_min is None:
            color_min = colors.min()
        if color_max is None:
            color_max = colors.max()
        
        colors_clamped = np.clip(colors, color_min, color_max)
        
        if color_max > color_min:
            colors_normalized = (colors_clamped - color_min) / (color_max - color_min)
        else:
            colors_normalized = np.zeros_like(colors_clamped)
        
        return colors_normalized
    
    def _get_lighting_settings(self, lighting_mode='soft_shadow', ambient_brightness=0.7):
        """Get optimized lighting settings for multi-view generation"""
        if lighting_mode == 'soft_shadow':
            return dict(
                ambient=ambient_brightness,  # Bright shadows for visibility
                diffuse=0.6,     # Moderate directional lighting
                specular=0.2,    # Subtle specular highlights
                roughness=0.8,   # Rougher surface for softer look
                fresnel=0.15     # Minimal fresnel
            )
        else:
            # Fallback to soft shadow settings
            return self._get_lighting_settings('soft_shadow', ambient_brightness)
    
    def _calculate_camera_position(self, azimuth_degrees, elevation_degrees, distance=2.0):
        """Calculate camera position for given angles"""
        azimuth_rad = math.radians(azimuth_degrees)
        elevation_rad = math.radians(elevation_degrees)
        
        x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = distance * math.sin(elevation_rad)
        
        return dict(x=x, y=y, z=z)
    
    def _calculate_surface_data(self):
        """Pre-calculate 3D surface data for optimization"""
        print("Pre-calculating 3D surface data...")
        
        try:
            # Create isosurface using marching cubes
            if np.count_nonzero(self.img_data_thresholded) == 0:
                data_for_surface = self.img_data
            else:
                data_for_surface = self.img_data_thresholded
            
            # Determine iso value
            unique_vals = np.unique(data_for_surface)
            if len(unique_vals) <= 2 and np.allclose(unique_vals, [0, 1]):
                iso_value = 0.5
            else:
                min_val = data_for_surface[data_for_surface > 0].min()
                max_val = data_for_surface.max()
                iso_value = min_val + (max_val - min_val) * 0.01
            
            # Generate surface
            verts, faces, normals, _ = measure.marching_cubes(
                data_for_surface, 
                level=iso_value,
                step_size=1
            )
            
            print(f"Pre-calculated surface: {len(verts)} vertices, {len(faces)} faces")
            
            # Calculate depth colors
            colors = self._calculate_depth_sum_colors(
                verts, normals, 
                depth_samples=15,  # Fixed depth samples for optimal results
                remove_outliers=True,
                outlier_percent=10.0
            )
            
            # Apply color mapping
            mapped_colors = self._apply_custom_color_mapping(colors)
            
            return {
                'vertices': verts,
                'faces': faces,
                'normals': normals,
                'colors': colors,
                'mapped_colors': mapped_colors
            }
            
        except Exception as e:
            print(f"Error pre-calculating surface data: {str(e)}")
            return None
    
    def _render_single_view(self, camera_position, image_size=800):
        """Render a single view using pre-calculated surface data"""
        if self._pre_calculated_surface is None:
            return None
        
        surface_data = self._pre_calculated_surface
        verts = surface_data['vertices']
        faces = surface_data['faces']
        mapped_colors = surface_data['mapped_colors']
        
        # Create custom red-to-green colorscale
        colorscale = [
            [0.0, 'rgb(255, 0, 0)'],      # Red for minimum values
            [0.5, 'rgb(255, 255, 0)'],    # Yellow for middle values  
            [1.0, 'rgb(0, 255, 0)']       # Green for maximum values
        ]
        
        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=mapped_colors,
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            opacity=1.0,
            name='Surface',
            lighting=self._get_lighting_settings('soft_shadow', 0.7),
            showscale=False,  # Hide color bar for clean images
        )
        
        fig = go.Figure(data=mesh)
        
        # Update layout for clean multi-view images
        fig.update_layout(
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                camera=dict(eye=camera_position),
                aspectmode='cube',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            width=image_size,
            height=image_size,
            title='',
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def generate_multi_view_images(self, rotation_step=90, elevation_step=30, 
                                 output_dir='output', image_size=800):
        """
        Generate multi-view 2D images with optimized settings
        
        Args:
            rotation_step (int): Step size in degrees for azimuth rotation
            elevation_step (int): Step size in degrees for elevation rotation  
            output_dir (str): Output directory for images
            image_size (int): Size of output images in pixels
        
        Returns:
            List of generated image file paths
        """
        print(f"Generating multi-view images with {rotation_step}° azimuth and {elevation_step}° elevation steps...")
        
        # Pre-calculate the 3D surface once
        self._pre_calculated_surface = self._calculate_surface_data()
        
        if self._pre_calculated_surface is None:
            print("Failed to pre-calculate surface data.")
            return []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate angles
        azimuth_angles = list(range(0, 360, rotation_step))
        elevation_angles = list(range(15, 91, elevation_step))
        total_images = len(azimuth_angles) * len(elevation_angles)
        image_paths = []
        
        # Generate base filename
        base_filename = os.path.basename(self.nifti_path)
        base_filename = base_filename.split('_')[0]
        
        print(f"Will generate {total_images} images ({len(azimuth_angles)} azimuth × {len(elevation_angles)} elevation angles)")
        
        image_count = 0
        for elevation in elevation_angles:
            for azimuth in azimuth_angles:
                image_count += 1
                print(f"Capturing view {image_count}/{total_images} - Azimuth: {azimuth}°, Elevation: {elevation}°")
                
                # Calculate camera position
                camera_pos = self._calculate_camera_position(azimuth, elevation)
                
                # Create figure
                fig = self._render_single_view(camera_pos, image_size)
                
                if fig is not None:
                    # Generate filename
                    filename = f"{base_filename}_az{azimuth:03d}_el{elevation:02d}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save image
                    try:
                        fig.write_image(filepath, width=image_size, height=image_size)
                        image_paths.append(filepath)
                        print(f"  Saved: {filepath}")
                    except Exception as e:
                        print(f"  Error saving {filepath}: {str(e)}")
                        if "kaleido" in str(e).lower():
                            print("  Install kaleido package: pip install kaleido")
                    finally:
                        # Clean up figure to prevent memory leaks
                        del fig
                        gc.collect()
        
        # Clean up pre-calculated data and force garbage collection
        self.cleanup_surface_data()
        self._force_garbage_collection()
        
        print(f"Multi-view capture complete! Generated {len(image_paths)} images in {output_dir}")
        return image_paths
    
    def cleanup_surface_data(self):
        """Clean up pre-calculated surface data"""
        if self._pre_calculated_surface is not None:
            self._pre_calculated_surface.clear()
            self._pre_calculated_surface = None
    
    def cleanup(self):
        """Clean up all resources and memory"""
        # Clean up image data
        if self.img_data is not None:
            del self.img_data
            self.img_data = None
        
        if self.img_data_thresholded is not None:
            del self.img_data_thresholded
            self.img_data_thresholded = None
        
        # Clean up nibabel image
        if self._nib_image is not None:
            del self._nib_image
            self._nib_image = None
        
        # Clean up surface data
        self.cleanup_surface_data()
        
        # Clean up other references
        self.affine = None
        self.header = None
        
        # Force garbage collection
        self._force_garbage_collection()
    
    def _force_garbage_collection(self):
        """Force garbage collection and report memory usage"""
        gc.collect()
        
        # Optional: Report current memory usage
        try:
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Current memory usage: {mem_usage:.1f} MB")
        except:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


class BatchProcessor:
    """Batch processor for processing entire datasets while preserving structure"""
    
    def __init__(self):
        self.processed_files = 0
        self.failed_files = 0
        self.total_files = 0
        self.failed_list = []
    
    def find_nifti_files(self, root_path: str) -> List[Tuple[str, str]]:
        """Find all NIfTI files in the dataset structure and their relative paths"""
        nifti_files = []
        root_path = Path(root_path)
        
        # Look for .nii.gz files recursively
        for nifti_file in root_path.rglob('*.nii.gz'):
            # Get relative path from root
            relative_path = nifti_file.relative_to(root_path)
            nifti_files.append((str(nifti_file), str(relative_path)))
        
        return nifti_files
    
    def create_output_structure(self, input_root: str, output_root: str) -> None:
        """Create output directory structure mirroring input structure"""
        input_path = Path(input_root)
        output_path = Path(output_root)
        
        # Create output root directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the directory structure (without files)
        for dir_path in input_path.rglob('*'):
            if dir_path.is_dir():
                relative_path = dir_path.relative_to(input_path)
                output_dir = output_path / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_file(self, input_file: str, output_dir: str, 
                          rotation_step: int = 45, elevation_step: int = 30,
                          image_size: int = 640, downsample: int = 1,
                          flip_upside_down: bool = True) -> bool:
        """Process a single NIfTI file and return success status"""
        generator = None
        try:
            print(f"Processing: {os.path.basename(input_file)}")
            
            # Check memory before processing
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize generator
            generator = MultiViewGenerator(input_file)
            
            # Preprocess data
            generator.preprocess_data(
                downsample_factor=downsample,
                flip_upside_down=flip_upside_down
            )
            
            # Generate multi-view images
            image_paths = generator.generate_multi_view_images(
                rotation_step=rotation_step,
                elevation_step=elevation_step,
                output_dir=output_dir,
                image_size=image_size
            )
            
            # Clean up generator
            generator.cleanup()
            generator = None
            
            # Force garbage collection after processing
            gc.collect()
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            print(f"  Memory used: {mem_after - mem_before:.1f} MB")
            
            if image_paths:
                print(f"  Success: Generated {len(image_paths)} images")
                return True
            else:
                print(f"  Failed: No images generated")
                return False
                
        except Exception as e:
            print(f"  Error processing {input_file}: {str(e)}")
            return False
        finally:
            # Ensure cleanup even if exception occurs
            if generator is not None:
                try:
                    generator.cleanup()
                except:
                    pass
            # Final garbage collection
            gc.collect()
    
    def process_batch(self, input_root: str, output_root: str,
                     rotation_step: int = 45, elevation_step: int = 30,
                     image_size: int = 640, downsample: int = 1,
                     flip_upside_down: bool = True) -> None:
        """Process entire dataset in batch mode"""
        print("=" * 80)
        print("BATCH PROCESSING MODE")
        print("=" * 80)
        print(f"Input dataset: {input_root}")
        print(f"Output dataset: {output_root}")
        print(f"Settings: rotation_step={rotation_step}°, elevation_step={elevation_step}°")
        print(f"Image size: {image_size}x{image_size}")
        print("=" * 80)
        
        # Find all NIfTI files
        print("Scanning for NIfTI files...")
        nifti_files = self.find_nifti_files(input_root)
        self.total_files = len(nifti_files)
        
        if self.total_files == 0:
            print("No NIfTI files found in the input directory.")
            return
        
        print(f"Found {self.total_files} NIfTI files to process")
        
        # Create output directory structure
        print("Creating output directory structure...")
        self.create_output_structure(input_root, output_root)
        
        # Process each file
        start_time = time.time()
        
        for i, (input_file, relative_path) in enumerate(nifti_files, 1):
            print(f"\n[{i}/{self.total_files}] Processing: {relative_path}")
            
            try:
                # Get the class directory (parent directory of the .nii.gz file)
                relative_path_obj = Path(relative_path)
                class_dir = relative_path_obj.parent  # This will be something like 'train/0' or 'val/1'
                output_class_dir = Path(output_root) / class_dir
                output_class_dir.mkdir(parents=True, exist_ok=True)
                
                # Validate input file before processing
                if not os.path.exists(input_file):
                    print(f"  ERROR: Input file not found: {input_file}")
                    self.failed_files += 1
                    self.failed_list.append(relative_path)
                    continue
                
                # Check file size (very small files might be corrupted)
                file_size = os.path.getsize(input_file)
                if file_size < 1024:  # Less than 1KB is suspicious
                    print(f"  WARNING: File size is very small ({file_size} bytes): {input_file}")
                
                # Process the file - images will go directly into the class directory
                success = self.process_single_file(
                    input_file=input_file,
                    output_dir=str(output_class_dir),
                    rotation_step=rotation_step,
                    elevation_step=elevation_step,
                    image_size=image_size,
                    downsample=downsample,
                    flip_upside_down=flip_upside_down
                )
                
            except Exception as processing_error:
                print(f"  ERROR: Unexpected error during processing setup: {str(processing_error)}")
                success = False
            
            if success:
                self.processed_files += 1
                print(f"  ✓ Successfully processed: {relative_path}")
            else:
                self.failed_files += 1
                self.failed_list.append(relative_path)
                print(f"  ✗ Failed to process: {relative_path}")
            
            # Progress update with memory monitoring
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / i
            remaining_files = self.total_files - i
            estimated_remaining_time = remaining_files * avg_time_per_file
            
            # Monitor memory usage
            try:
                process = psutil.Process(os.getpid())
                mem_usage = process.memory_info().rss / 1024 / 1024  # MB
                mem_info = f", Memory: {mem_usage:.1f} MB"
            except:
                mem_info = ""
            
            print(f"  Progress: {i}/{self.total_files} files processed")
            print(f"  Success: {self.processed_files}, Failed: {self.failed_files}")
            print(f"  Elapsed: {elapsed_time:.1f}s, ETA: {estimated_remaining_time:.1f}s{mem_info}")
            
            # Force garbage collection every 10 files to prevent memory buildup
            if i % 10 == 0:
                print(f"  Performing memory cleanup...")
                gc.collect()
                
            # Check for high memory usage and warn
            try:
                mem_usage_gb = process.memory_info().rss / 1024 / 1024 / 1024
                if mem_usage_gb > 8.0:  # Warn if using more than 8GB
                    print(f"  WARNING: High memory usage detected: {mem_usage_gb:.1f} GB")
            except:
                pass
        
        # Final cleanup and summary
        gc.collect()  # Final garbage collection
        total_time = time.time() - start_time
        self._print_final_summary(total_time)
    
    def _print_final_summary(self, total_time: float) -> None:
        """Print final processing summary with memory info"""
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total files processed: {self.total_files}")
        print(f"Successfully processed: {self.processed_files}")
        print(f"Failed: {self.failed_files}")
        print(f"Success rate: {(self.processed_files/self.total_files)*100:.1f}%")
        print(f"Total processing time: {total_time:.1f} seconds")
        print(f"Average time per file: {total_time/self.total_files:.1f} seconds")
        
        # Final memory report
        try:
            process = psutil.Process(os.getpid())
            final_mem = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Final memory usage: {final_mem:.1f} MB")
        except:
            pass
        
        if self.failed_list:
            print("\nFailed files:")
            for failed_file in self.failed_list:
                print(f"  - {failed_file}")
            print("\n💡 Troubleshooting tips for failed files:")
            print("  - Check if files are corrupted or incomplete")
            print("  - Verify file permissions")
            print("  - Ensure sufficient disk space in output directory")
            print("  - Try processing failed files individually for detailed error messages")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='3D NIfTI to Multi-view 2D Image Generator',
        epilog='''
Examples:
  # Single file processing
  %(prog)s input.nii.gz output_folder
  %(prog)s input.nii.gz output_folder --rotation_step 45
  
  # Batch processing (preserves dataset structure)
  %(prog)s --batch input_dataset_root output_dataset_root
  %(prog)s --batch input_dataset_root output_dataset_root --rotation_step 30
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Batch processing flag
    parser.add_argument('--batch', action='store_true',
                       help='Enable batch processing mode')
    
    # Input/output arguments (conditional based on batch mode)
    parser.add_argument('input', help='Input .nii.gz file (single mode) or dataset root directory (batch mode)')
    parser.add_argument('output', help='Output directory for images (single mode) or dataset root directory (batch mode)')
    
    # Optional parameters with optimized defaults
    parser.add_argument('--rotation_step', type=int, default=45,
                       help='Azimuth rotation step in degrees (default: 45)')
    parser.add_argument('--elevation_step', type=int, default=30,
                       help='Elevation rotation step in degrees (default: 30)')
    parser.add_argument('--image_size', type=int, default=640,
                       help='Size of output images in pixels (default: 640)')
    parser.add_argument('--downsample', type=int, default=1,
                       help='Downsampling factor for performance (default: 1)')
    parser.add_argument('--no_flip', action='store_true',
                       help='Disable upside-down flip (default: flipped)')
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        input_type = "directory" if args.batch else "file"
        print(f"Error: Input {input_type} '{args.input}' not found.")
        return 1
    
    try:
        if args.batch:
            # Batch processing mode
            processor = BatchProcessor()
            processor.process_batch(
                input_root=args.input,
                output_root=args.output,
                rotation_step=args.rotation_step,
                elevation_step=args.elevation_step,
                image_size=args.image_size,
                downsample=args.downsample,
                flip_upside_down=not args.no_flip
            )
            return 0 if processor.failed_files == 0 else 1
            
        else:
            # Single file processing mode
            print("=" * 60)
            print("3D NIfTI to Multi-view 2D Image Generator")
            print("=" * 60)
            print(f"Input file: {args.input}")
            print(f"Output directory: {args.output}")
            print(f"Rotation step: {args.rotation_step}°")
            print(f"Elevation step: {args.elevation_step}°")
            print(f"Image size: {args.image_size}x{args.image_size}")
            print("=" * 60)
            
            # Initialize generator
            print("Loading NIfTI file...")
            generator = MultiViewGenerator(args.input)
            
            print("Preprocessing data...")
            generator.preprocess_data(
                downsample_factor=args.downsample,
                flip_upside_down=not args.no_flip
            )
            
            # Generate multi-view images
            image_paths = generator.generate_multi_view_images(
                rotation_step=args.rotation_step,
                elevation_step=args.elevation_step,
                output_dir=args.output,
                image_size=args.image_size
            )
            
            if image_paths:
                print("=" * 60)
                print(f"SUCCESS: Generated {len(image_paths)} multi-view images!")
                print(f"Output directory: {args.output}")
                print("=" * 60)
                return 0
            else:
                print("ERROR: Failed to generate multi-view images.")
                print("Make sure you have the 'kaleido' package installed: pip install kaleido")
                return 1
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())