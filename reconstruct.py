#!/usr/bin/env python
# Integrated VGGT + Point Cloud Registration with Overlapping Image Batches

import os
import cv2
import torch
import numpy as np
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import argparse
import copy
import trimesh
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# -------------------------------------------------------------------------
# Overlapping Batch Processing Functions
# -------------------------------------------------------------------------


def improve_camera_based_initialization(source_cameras, target_cameras, source_images, target_images):
    """
    Create a better initial transformation using camera poses to prevent bottom-to-top matching issues
    """
    print("Finding optimal camera alignment...")
    
    # Find overlapping camera frames if any
    source_filenames = [os.path.splitext(os.path.basename(path))[0] for path in source_images]
    target_filenames = [os.path.splitext(os.path.basename(path))[0] for path in target_images]
    common_files = set(source_filenames) & set(target_filenames)
    
    # Get indices in each batch
    source_overlap_idx = [source_filenames.index(f) for f in common_files if f in source_filenames]
    target_overlap_idx = [target_filenames.index(f) for f in common_files if f in target_filenames]
    
    # Try multiple alignment strategies and choose the best one
    alignment_options = []
    
    # Option 1: Use overlapping cameras if available
    if len(source_overlap_idx) >= 3:
        print(f"Strategy 1: Using {len(source_overlap_idx)} explicitly overlapping cameras")
        source_pts = source_cameras[source_overlap_idx]
        target_pts = target_cameras[target_overlap_idx]
        
        # Calculate transformation
        transform, error = compute_rigid_transform(source_pts, target_pts)
        alignment_options.append(("Overlapping Cameras", transform, error, source_overlap_idx, target_overlap_idx))
    
    # Option 2: Try start-to-end alignment (last cameras of target to first cameras of source)
    num_match = min(8, min(len(source_cameras), len(target_cameras)))
    source_start_idx = list(range(num_match))
    target_end_idx = list(range(len(target_cameras) - num_match, len(target_cameras)))
    
    print(f"Strategy 2: Matching last {num_match} cameras of target to first {num_match} cameras of source")
    source_pts = source_cameras[:num_match]
    target_pts = target_cameras[-num_match:]
    transform, error = compute_rigid_transform(source_pts, target_pts)
    alignment_options.append(("Start-to-End", transform, error, source_start_idx, target_end_idx))
    
    # Option 3: Try end-to-start alignment (first cameras of target to last cameras of source)
    source_end_idx = list(range(len(source_cameras) - num_match, len(source_cameras)))
    target_start_idx = list(range(num_match))
    
    print(f"Strategy 3: Matching first {num_match} cameras of target to last {num_match} cameras of source")
    source_pts = source_cameras[-num_match:]
    target_pts = target_cameras[:num_match]
    transform, error = compute_rigid_transform(source_pts, target_pts)
    alignment_options.append(("End-to-Start", transform, error, source_end_idx, target_start_idx))
    
    # Choose the best alignment based on error
    alignment_options.sort(key=lambda x: x[2])
    best_alignment = alignment_options[0]
    
    print(f"Selected alignment strategy: {best_alignment[0]} with error: {best_alignment[2]:.6f}")
    return best_alignment[1], best_alignment[3], best_alignment[4]


def compute_rigid_transform(source_pts, target_pts):
    """Calculate optimal rigid transform between point sets and return error metric"""
    # Calculate centroids
    source_centroid = np.mean(source_pts, axis=0)
    target_centroid = np.mean(target_pts, axis=0)
    
    # Center the points
    source_centered = source_pts - source_centroid
    target_centered = target_pts - target_centroid
    
    # Compute rotation using SVD
    H = np.dot(source_centered.T, target_centered)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Ensure proper rotation matrix (det=1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Translation
    t = target_centroid - np.dot(R, source_centroid)
    
    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    
    # Calculate error metric (average distance after transform)
    transformed_source = np.dot(source_centered, R.T) + target_centroid
    error = np.mean(np.linalg.norm(transformed_source - target_pts, axis=1))
    
    return transformation, error


def visualize_registration_overlap(source_pcd, target_pcd, source_cameras, target_cameras, 
                                  source_overlap_idx, target_overlap_idx, transformation, output_path):
    """
    Create a visualization GLB showing both point clouds and their camera positions
    """
    # Transform source to align with target
    source_transformed = copy.deepcopy(source_pcd)
    source_transformed.transform(transformation)
    
    # Color the point clouds differently
    source_points = np.asarray(source_transformed.points)
    source_colors = np.ones((len(source_points), 3)) * [1.0, 0.7, 0.0]  # Yellow for source
    
    target_points = np.asarray(target_pcd.points)
    target_colors = np.ones((len(target_points), 3)) * [0.0, 0.65, 0.93]  # Blue for target
    
    # Combine the point clouds
    all_points = np.vstack((source_points, target_points))
    all_colors = np.vstack((source_colors, target_colors))
    all_colors = (all_colors * 255).astype(np.uint8)
    
    # Create trimesh scene
    scene = trimesh.Scene()
    
    # Add the combined point cloud
    cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
    scene.add_geometry(cloud)
    
    # Transform source cameras
    transformed_source_cameras = []
    for cam in source_cameras:
        # Convert to homogeneous coordinates
        cam_h = np.ones(4)
        cam_h[:3] = cam
        # Transform
        cam_transformed = transformation @ cam_h
        transformed_source_cameras.append(cam_transformed[:3])
    
    # Add camera positions as spheres
    for i, cam_pos in enumerate(transformed_source_cameras):
        # Red for overlapping source cameras, orange for others
        color = [255, 0, 0, 255] if i in source_overlap_idx else [255, 165, 0, 255]
        sphere = trimesh.primitives.Sphere(radius=0.05, center=cam_pos)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)
    
    for i, cam_pos in enumerate(target_cameras):
        # Green for overlapping target cameras, cyan for others
        color = [0, 255, 0, 255] if i in target_overlap_idx else [0, 255, 255, 255]
        sphere = trimesh.primitives.Sphere(radius=0.05, center=cam_pos)
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)
    
    # Add camera trajectory lines using cylinders instead of paths
    # Source trajectory (red)
    if len(transformed_source_cameras) >= 2:
        for i in range(len(transformed_source_cameras) - 1):
            p1 = transformed_source_cameras[i]
            p2 = transformed_source_cameras[i+1]
            
            # Create a cylinder between consecutive camera positions
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length > 0:  # Avoid zero-length cylinders
                # Create a cylinder
                radius = 0.01  # Cylinder radius
                center = (p1 + p2) / 2
                
                # Calculate rotation to align cylinder with direction
                direction_norm = direction / length
                z_axis = np.array([0, 0, 1])
                
                if np.allclose(direction_norm, z_axis) or np.allclose(direction_norm, -z_axis):
                    rotation = np.eye(3)
                else:
                    # Find rotation from z-axis to direction
                    cross = np.cross(z_axis, direction_norm)
                    sin_angle = np.linalg.norm(cross)
                    cos_angle = np.dot(z_axis, direction_norm)
                    
                    # Skew-symmetric cross-product matrix
                    K = np.array([[0, -cross[2], cross[1]],
                                [cross[2], 0, -cross[0]],
                                [-cross[1], cross[0], 0]])
                    
                    # Rodrigues formula
                    if sin_angle > 0:
                        rotation = np.eye(3) + K + K @ K * (1 - cos_angle) / (sin_angle ** 2)
                    else:
                        rotation = np.eye(3)
                
                # Create cylinder
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = center
                
                cylinder = trimesh.creation.cylinder(radius=radius, height=length)
                cylinder.apply_transform(transform)
                cylinder.visual.face_colors = [255, 0, 0, 100]  # Red, semi-transparent
                scene.add_geometry(cylinder)
    
    # Target trajectory (green)
    if len(target_cameras) >= 2:
        for i in range(len(target_cameras) - 1):
            p1 = target_cameras[i]
            p2 = target_cameras[i+1]
            
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length > 0:
                radius = 0.01
                center = (p1 + p2) / 2
                
                direction_norm = direction / length
                z_axis = np.array([0, 0, 1])
                
                if np.allclose(direction_norm, z_axis) or np.allclose(direction_norm, -z_axis):
                    rotation = np.eye(3)
                else:
                    cross = np.cross(z_axis, direction_norm)
                    sin_angle = np.linalg.norm(cross)
                    cos_angle = np.dot(z_axis, direction_norm)
                    
                    K = np.array([[0, -cross[2], cross[1]],
                                [cross[2], 0, -cross[0]],
                                [-cross[1], cross[0], 0]])
                    
                    if sin_angle > 0:
                        rotation = np.eye(3) + K + K @ K * (1 - cos_angle) / (sin_angle ** 2)
                    else:
                        rotation = np.eye(3)
                
                transform = np.eye(4)
                transform[:3, :3] = rotation
                transform[:3, 3] = center
                
                cylinder = trimesh.creation.cylinder(radius=radius, height=length)
                cylinder.apply_transform(transform)
                cylinder.visual.face_colors = [0, 255, 0, 100]  # Green, semi-transparent
                scene.add_geometry(cylinder)
    
    # Export the scene
    scene.export(output_path)
    print(f"Registration visualization saved to {output_path}")
    
    return scene

def create_overlapping_batches(image_paths, batch_size=16, overlap_size=8):
    """
    Create overlapping batches of sequential images
    
    Args:
        image_paths: List of paths to input images (already sorted)
        batch_size: Number of images per batch
        overlap_size: Number of overlapping images between consecutive batches
        
    Returns:
        List of image path batches
    """
    if batch_size <= overlap_size:
        raise ValueError(f"Batch size ({batch_size}) must be larger than overlap size ({overlap_size})")
    
    if len(image_paths) <= batch_size:
        return [image_paths]
    
    batches = []
    step_size = batch_size - overlap_size
    
    for i in range(0, len(image_paths), step_size):
        batch_end = min(i + batch_size, len(image_paths))
        batch = image_paths[i:batch_end]
        batches.append(batch)
        
        # If we can't form a full batch anymore, stop
        if batch_end == len(image_paths):
            break
    
    # Ensure the last batch is at least half the batch_size to avoid tiny batches
    if len(batches) > 1 and len(batches[-1]) < batch_size // 2:
        # Extend the last batch by including more images from the previous batch
        batches[-2].extend(batches[-1])
        batches.pop()
    
    return batches


def setup_model():
    """Initialize and return the VGGT model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing and loading VGGT model on {device}...")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)
    
    return model


def process_image_batch(image_batch, model, batch_id, output_dir):
    """
    Process a batch of images and save results
    
    Args:
        image_batch: List of paths to images in this batch
        model: VGGT model instance
        batch_id: ID for this batch
        output_dir: Directory to save batch results
        
    Returns:
        Dictionary with batch predictions and paths
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create batch directory
    batch_dir = os.path.join(output_dir, f"batch_{batch_id:03d}")
    batch_images_dir = os.path.join(batch_dir, "images")
    os.makedirs(batch_images_dir, exist_ok=True)
    
    # Copy images to batch directory for reference
    batch_image_paths = []
    for i, path in enumerate(image_batch):
        dst_path = os.path.join(batch_images_dir, f"{i:04d}{os.path.splitext(path)[1]}")
        shutil.copy(path, dst_path)
        batch_image_paths.append(dst_path)
    
    # Process images with VGGT
    print(f"Processing batch {batch_id} with {len(batch_image_paths)} images...")
    
    # Load and preprocess images
    images = load_and_preprocess_images(batch_image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Run inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images)
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    # Convert tensors to numpy arrays
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    # Generate world points from depth map
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Save predictions to NPZ file
    npz_path = os.path.join(batch_dir, "predictions.npz")
    np.savez(npz_path, **predictions)
    
    # Generate individual batch GLB
    glb_path = os.path.join(batch_dir, "scene.glb")
    glbscene = predictions_to_glb(
        predictions, 
        conf_thres=50.0,
        filter_by_frames="All",
        mask_black_bg=True,
        mask_white_bg=True,
        show_cam=True,
        mask_sky=False,
        target_dir=batch_dir,
        prediction_mode="Depthmap and Camera Branch"
    )
    glbscene.export(file_obj=glb_path)
    
    result = {
        "predictions": predictions,
        "batch_dir": batch_dir,
        "glb_path": glb_path,
        "npz_path": npz_path,
        "image_paths": batch_image_paths,
        "batch_id": batch_id
    }
    
    # Clean CUDA memory
    torch.cuda.empty_cache()
    
    return result


# -------------------------------------------------------------------------
# Point Cloud Extraction and Registration
# -------------------------------------------------------------------------
def extract_point_cloud_from_predictions(predictions, conf_threshold=50.0, voxel_size=0.02, mask_bg=True):
    """Extract Open3D point cloud from predictions with filtering"""
    # Get points and confidence
    if "world_points" in predictions:
        points = predictions["world_points"]
        conf = predictions.get("world_points_conf", np.ones_like(points[..., 0]))
    else:
        points = predictions["world_points_from_depth"]
        conf = predictions.get("depth_conf", np.ones_like(points[..., 0]))
    
    # Get images
    images = predictions["images"]
    
    # Handle different image formats
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # NHWC format
        colors_rgb = images
    
    # Reshape data
    vertices_3d = points.reshape(-1, 3)
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)
    conf = conf.reshape(-1)
    
    # Apply confidence threshold
    if conf_threshold > 0:
        conf_threshold_value = np.percentile(conf, conf_threshold)
        conf_mask = (conf >= conf_threshold_value) & (conf > 1e-5)
    else:
        conf_mask = np.ones_like(conf, dtype=bool)
    
    # Filter background
    if mask_bg:
        # Filter black background
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        # Filter white background
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & black_bg_mask & white_bg_mask
    
    # Apply masks
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
    
    # Downsample to make processing more efficient
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Get camera positions
    extrinsics = predictions["extrinsic"]
    camera_centers = extract_camera_centers(extrinsics)
    
    return pcd, camera_centers


def extract_camera_centers(extrinsics):
    """Extract camera centers from extrinsic matrices"""
    centers = []
    for ext in extrinsics:
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :4] = ext
        
        # Get camera center in world coordinates (-R^T * t)
        R = T[:3, :3]
        t = T[:3, 3]
        center = -np.dot(R.T, t)
        centers.append(center)
    
    return np.array(centers)


def find_overlapping_cameras(batch1_cameras, batch2_cameras, batch1_images, batch2_images):
    """
    Identify overlapping cameras between batches based on image filenames
    
    Returns:
        Indices of overlapping cameras in each batch
    """
    # Extract base filenames without directories and extensions
    batch1_filenames = [os.path.splitext(os.path.basename(path))[0] for path in batch1_images]
    batch2_filenames = [os.path.splitext(os.path.basename(path))[0] for path in batch2_images]
    
    # Find common filenames
    common_files = set(batch1_filenames) & set(batch2_filenames)
    
    # Get indices in each batch
    batch1_indices = [batch1_filenames.index(f) for f in common_files if f in batch1_filenames]
    batch2_indices = [batch2_filenames.index(f) for f in common_files if f in batch2_filenames]
    
    return batch1_indices, batch2_indices


def register_consecutive_point_clouds(source_pcd, target_pcd, source_cameras, target_cameras, 
                                     source_images, target_images, output_dir=None, batch_id=None):
    """
    Register two point clouds from consecutive batches with improved initialization
    """
    print("Registering consecutive point clouds...")
    
    # 1. Better initialization using camera poses
    initial_transformation, source_overlap_idx, target_overlap_idx = improve_camera_based_initialization(
        source_cameras, target_cameras, source_images, target_images
    )
    
    # 2. Visualize pre-registration alignment
    if output_dir is not None and batch_id is not None:
        pre_reg_path = os.path.join(output_dir, f"pre_registration_batch_{batch_id}.glb")
        visualize_registration_overlap(
            source_pcd, target_pcd, source_cameras, target_cameras,
            source_overlap_idx, target_overlap_idx, initial_transformation, pre_reg_path
        )
    
    # 3. Apply initial transformation and proceed with ICP refinement
    source_aligned = copy.deepcopy(source_pcd)
    source_aligned.transform(initial_transformation)
    
    # Prepare for ICP
    source_aligned.estimate_normals()
    target_pcd.estimate_normals()
    
    # Try different refinement methods
    try:
        print("Running colored ICP for fine alignment...")
        reg_p2p = o3d.pipelines.registration.registration_colored_icp(
            source_aligned, target_pcd, 0.05, 
            np.eye(4),  # Start from current alignment
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        print(f"ICP fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
        
        # Only use ICP refinement if it significantly improves the alignment
        if reg_p2p.fitness > 0.3:  # Adjust threshold as needed
            final_transformation = reg_p2p.transformation @ initial_transformation
        else:
            print("ICP did not improve alignment significantly, using initial transformation")
            final_transformation = initial_transformation
    except Exception as e:
        print(f"ICP refinement failed: {str(e)}")
        final_transformation = initial_transformation
    
    # 4. Visualize post-registration alignment
    if output_dir is not None and batch_id is not None:
        post_reg_path = os.path.join(output_dir, f"post_registration_batch_{batch_id}.glb")
        visualize_registration_overlap(
            source_pcd, target_pcd, source_cameras, target_cameras,
            source_overlap_idx, target_overlap_idx, final_transformation, post_reg_path
        )
    
    return final_transformation


def register_all_point_clouds(batch_results, output_dir, voxel_size=0.02):
    """
    Register all point clouds with better initialization and visualization
    """
    point_clouds = []
    camera_positions = []
    transformations = []
    
    # Extract point cloud from each batch
    for batch_result in batch_results:
        predictions = batch_result["predictions"]
        pcd, cameras = extract_point_cloud_from_predictions(
            predictions, conf_threshold=30.0, voxel_size=voxel_size, mask_bg=True
        )
        point_clouds.append(pcd)
        camera_positions.append(cameras)
    
    # Register consecutive point clouds
    for i in range(1, len(point_clouds)):
        source_pcd = point_clouds[i]
        target_pcd = point_clouds[i-1]
        source_cameras = camera_positions[i]
        target_cameras = camera_positions[i-1]
        source_images = batch_results[i]["image_paths"]
        target_images = batch_results[i-1]["image_paths"]
        
        # Pass both output_dir and batch_id
        transformation = register_consecutive_point_clouds(
            source_pcd, target_pcd, source_cameras, target_cameras,
            source_images, target_images, output_dir, i
        )
        
        transformations.append(transformation)
    
    # Apply transformations to align all point clouds
    aligned_point_clouds = [point_clouds[0]]  # First cloud is reference
    
    for i in range(1, len(point_clouds)):
        # Calculate cumulative transformation
        cumulative_transform = np.eye(4)
        for j in range(i-1, -1, -1):
            cumulative_transform = transformations[j] @ cumulative_transform
        
        # Apply transformation
        transformed_pcd = copy.deepcopy(point_clouds[i])
        transformed_pcd.transform(cumulative_transform)
        aligned_point_clouds.append(transformed_pcd)
    
    return aligned_point_clouds, transformations


def merge_aligned_point_clouds(aligned_point_clouds, voxel_size=0.02):
    """
    Merge aligned point clouds into a single consistent model
    """
    if not aligned_point_clouds:
        return None
    
    # Merge all aligned clouds
    merged_cloud = o3d.geometry.PointCloud()
    for pcd in aligned_point_clouds:
        merged_cloud += pcd
    
    # Clean and optimize the merged point cloud
    print("Cleaning and optimizing merged point cloud...")
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size)
    merged_cloud, _ = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return merged_cloud


# -------------------------------------------------------------------------
# Main Reconstruction Pipeline
# -------------------------------------------------------------------------
def reconstruct_from_images_with_overlap(image_dir, output_path, batch_size=8, overlap_size=2,
                                       conf_threshold=50.0, voxel_size=0.02):
    """
    Complete end-to-end reconstruction pipeline with overlapping image batches
    
    Args:
        image_dir: Directory containing input images
        output_path: Path to save the output merged GLB file
        batch_size: Number of images per batch
        overlap_size: Number of overlapping images between batches
        conf_threshold: Confidence threshold for filtering points (0-100)
        voxel_size: Voxel size for downsampling point clouds
    """
    start_time = time.time()
    
    # Create output directory for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"reconstructions/reconstruction_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images in directory
    image_paths = []
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    
    # Sort images to ensure sequential processing
    image_paths = sorted(image_paths)
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Create overlapping batches
    batches = create_overlapping_batches(image_paths, batch_size, overlap_size)
    print(f"Created {len(batches)} overlapping batches")
    
    # Initialize model
    model = setup_model()
    
    # Process each batch
    batch_results = []
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)} with {len(batch)} images")
        batch_result = process_image_batch(batch, model, i, output_dir)
        batch_results.append(batch_result)
        
        # Save memory by removing large prediction data after extraction
        if "predictions" in batch_result:
            del batch_result["predictions"]
            gc.collect()
    
    # Register and merge point clouds
    print("\nRegistering and merging point clouds...")
    
    # Load predictions again from NPZ files to save memory during processing
    for i, result in enumerate(batch_results):
        predictions = dict(np.load(result["npz_path"], allow_pickle=True))
        batch_results[i]["predictions"] = predictions
    
    # Register point clouds - FIXED LINE BELOW
    aligned_point_clouds, transformations = register_all_point_clouds(batch_results, output_dir, voxel_size)
    # Register point cloud
    
    # Merge point clouds
    merged_cloud = merge_aligned_point_clouds(aligned_point_clouds, voxel_size)
    
    # Convert to trimesh scene
    print("Creating final trimesh scene...")
    vertices = np.asarray(merged_cloud.points)
    colors = (np.asarray(merged_cloud.colors) * 255).astype(np.uint8)
    
    scene = trimesh.Scene()
    point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
    scene.add_geometry(point_cloud)
    
    # Export merged result to GLB
    if not output_path.endswith('.glb'):
        output_path += '.glb'
    
    merged_output_path = os.path.join(output_dir, os.path.basename(output_path))
    print(f"Exporting merged scene to {merged_output_path}...")
    scene.export(file_obj=merged_output_path)
    
    # Also copy to requested output location
    shutil.copy(merged_output_path, output_path)
    
    end_time = time.time()
    print(f"\nReconstruction completed in {end_time - start_time:.2f} seconds")
    print(f"Final point cloud has {len(vertices)} points")
    print(f"\nResults:")
    print(f"- Individual batch GLBs in subdirectories of {output_dir}")
    print(f"- Final merged GLB at {output_path}")
    print(f"- Working files in {output_dir}")
    
    return output_path, output_dir


# -------------------------------------------------------------------------
# Main Function and CLI Interface
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="3D reconstruction from sequential images with overlapping batches"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, 
        help="Directory containing input images in sequential order"
    )
    parser.add_argument(
        "--output", type=str, default="merged_scene.glb", 
        help="Output GLB file path for the merged scene"
    )
    parser.add_argument(
        "--batch_size", type=int, default=28,
        help="Number of images per batch"
    )
    parser.add_argument(
        "--overlap_size", type=int, default=16,
        help="Number of overlapping images between consecutive batches"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=20.0, 
        help="Confidence threshold percentage for filtering points (0-100)"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.02,
        help="Voxel size for point cloud downsampling"
    )
    
    args = parser.parse_args()
    
    output_path, output_dir = reconstruct_from_images_with_overlap(
        image_dir=args.image_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        overlap_size=args.overlap_size,
        conf_threshold=args.conf_threshold,
        voxel_size=args.voxel_size
    )
    
    print(f"Reconstruction complete! Output saved to {output_path}")
    print(f"Working directory with individual GLBs: {output_dir}")


if __name__ == "__main__":
    main()