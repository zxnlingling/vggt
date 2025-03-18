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
        conf_thres=90.0,
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
                                     source_images, target_images):
    """
    Register two point clouds from consecutive batches using both overlapping cameras
    and point cloud features
    
    Returns:
        Transformation matrix to align source to target
    """
    print("Registering consecutive point clouds...")
    
    # 1. Find overlapping cameras
    source_overlap_idx, target_overlap_idx = find_overlapping_cameras(
        source_cameras, target_cameras, source_images, target_images
    )
    
    if len(source_overlap_idx) < 2:
        print("Warning: Less than 2 overlapping cameras found, alignment may be inaccurate")
        # Fall back to using all cameras if not enough overlap
        source_pts = source_cameras
        target_pts = target_cameras[:len(source_pts)] if len(target_cameras) > len(source_pts) else target_cameras
    else:
        # Use overlapping cameras for initial alignment
        source_pts = source_cameras[source_overlap_idx]
        target_pts = target_cameras[target_overlap_idx]
    
    print(f"Found {len(source_overlap_idx)} overlapping cameras for alignment")
    
    # 2. Initial alignment using camera positions
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
    
    # Initial transformation matrix
    initial_transformation = np.eye(4)
    initial_transformation[:3, :3] = R
    initial_transformation[:3, 3] = t
    
    # 3. Apply initial transformation to source point cloud
    source_aligned = copy.deepcopy(source_pcd)
    source_aligned.transform(initial_transformation)
    
    # 4. Refine alignment with Colored ICP
    # Prepare for colored ICP
    source_aligned.estimate_normals()
    target_pcd.estimate_normals()
    
    # Run Colored ICP
    print("Running colored ICP for fine alignment...")
    reg_p2p = o3d.pipelines.registration.registration_colored_icp(
        source_aligned, target_pcd, 0.05, 
        initial_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    # Final transformation matrix (combines initial and refinement)
    final_transformation = reg_p2p.transformation @ initial_transformation
    
    print(f"Registration complete with fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
    
    return final_transformation


def register_all_point_clouds(batch_results, voxel_size=0.02):
    """
    Register all point clouds from overlapping batches
    
    Returns:
        List of transformation matrices and list of point clouds
    """
    point_clouds = []
    camera_positions = []
    transformations = []
    
    # Extract point cloud from each batch
    for batch_result in batch_results:
        predictions = batch_result["predictions"]
        pcd, cameras = extract_point_cloud_from_predictions(
            predictions, conf_threshold=50.0, voxel_size=voxel_size, mask_bg=True
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
        
        transformation = register_consecutive_point_clouds(
            source_pcd, target_pcd, source_cameras, target_cameras,
            source_images, target_images
        )
        
        transformations.append(transformation)
    
    # Apply transformations to point clouds
    aligned_point_clouds = [point_clouds[0]]  # First cloud is reference
    
    # Forward transformations - transform each cloud to first cloud's reference frame
    for i in range(1, len(point_clouds)):
        # Initialize with identity transformation
        cumulative_transform = np.eye(4)
        
        # Compose transformations from current cloud back to the first cloud
        for j in range(i-1, -1, -1):
            cumulative_transform = transformations[j] @ cumulative_transform
        
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
    
    # Register point clouds
    aligned_point_clouds, transformations = register_all_point_clouds(batch_results, voxel_size)
    
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
        "--batch_size", type=int, default=30,
        help="Number of images per batch"
    )
    parser.add_argument(
        "--overlap_size", type=int, default=20,
        help="Number of overlapping images between consecutive batches"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=90.0, 
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