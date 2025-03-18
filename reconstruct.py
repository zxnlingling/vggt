import numpy as np
import open3d as o3d
import copy
import trimesh
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def merge_point_clouds(prediction_sets, voxel_size=0.02):
    """
    Register and merge multiple point clouds from overlapping image sets
    
    Args:
        prediction_sets: List of prediction dictionaries containing point clouds and camera parameters
        voxel_size: Resolution for downsampling and registration
        
    Returns:
        merged_scene: Final trimesh.Scene with merged point cloud
    """
    # 1. Extract point clouds and camera positions from each prediction set
    point_clouds = []
    camera_positions = []
    
    for i, predictions in enumerate(prediction_sets):
        # Extract point cloud data
        if "world_points" in predictions:
            points = predictions["world_points"]
        else:
            points = predictions["world_points_from_depth"]
        
        # Get confidence values if available
        if "world_points_conf" in predictions:
            conf = predictions["world_points_conf"]
        else:
            conf = np.ones_like(points[..., 0])
            
        # Get RGB images
        images = predictions["images"]
        
        # Extract camera matrices
        extrinsics = predictions["extrinsic"]
        
        # Create colored point cloud
        vertices, colors = extract_colored_point_cloud(points, images, conf, threshold=50.0)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0,1]
        
        # Downsample to speed up registration
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Extract camera centers from extrinsic matrices
        cam_centers = extract_camera_centers(extrinsics)
        
        point_clouds.append(pcd_down)
        camera_positions.append(cam_centers)
        
        print(f"Point cloud {i}: {len(pcd_down.points)} points, {len(cam_centers)} cameras")
    
    # 2. Register point clouds using both camera positions and point features
    transformations = register_point_clouds(point_clouds, camera_positions)
    
    # 3. Transform and merge point clouds
    merged_cloud = o3d.geometry.PointCloud()
    
    for i, pcd in enumerate(point_clouds):
        # Apply transformation
        if i > 0:  # First cloud is reference
            pcd_transformed = copy.deepcopy(pcd)
            pcd_transformed.transform(transformations[i-1])
            merged_cloud += pcd_transformed
        else:
            merged_cloud += pcd
    
    # 4. Clean and optimize final point cloud
    print("Cleaning merged point cloud...")
    merged_cloud = merged_cloud.voxel_down_sample(voxel_size)
    merged_cloud, _ = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 5. Create trimesh scene for output
    merged_scene = trimesh.Scene()
    
    # Convert merged point cloud to trimesh
    vertices = np.asarray(merged_cloud.points)
    colors = (np.asarray(merged_cloud.colors) * 255).astype(np.uint8)
    merged_trimesh = trimesh.PointCloud(vertices=vertices, colors=colors)
    merged_scene.add_geometry(merged_trimesh)
    
    return merged_scene


def extract_colored_point_cloud(points, images, confidence, threshold=50.0):
    """Extract points and colors based on confidence threshold"""
    S, H, W, _ = points.shape
    vertices_3d = points.reshape(-1, 3)
    
    # Handle different image formats
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # NHWC format
        colors_rgb = images
    
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)
    conf = confidence.reshape(-1)
    
    # Apply confidence threshold
    if threshold > 0:
        conf_threshold = np.percentile(conf, threshold)
        mask = (conf >= conf_threshold) & (conf > 1e-5)
        vertices_3d = vertices_3d[mask]
        colors_rgb = colors_rgb[mask]
    
    return vertices_3d, colors_rgb


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


def register_point_clouds(point_clouds, camera_positions):
    """
    Register point clouds using both camera positions and ICP
    
    Returns:
        List of transformation matrices to align each point cloud to the first one
    """
    if len(point_clouds) <= 1:
        return []
    
    transformations = []
    target_cloud = point_clouds[0]
    target_cameras = camera_positions[0]
    
    # Process each subsequent point cloud
    for i in range(1, len(point_clouds)):
        source_cloud = point_clouds[i]
        source_cameras = camera_positions[i]
        
        # 1. Initial alignment using camera positions (if sequential)
        initial_transformation = estimate_transformation_from_cameras(source_cameras, target_cameras)
        
        # Apply initial transformation
        source_aligned = copy.deepcopy(source_cloud)
        source_aligned.transform(initial_transformation)
        
        # 2. Fine registration using Colored ICP
        # Prepare for colored ICP
        source_aligned.estimate_normals()
        target_cloud.estimate_normals()
        
        # Run Colored ICP (slower but more accurate for colored point clouds)
        print(f"Running colored ICP for point cloud {i}...")
        reg_p2p = o3d.pipelines.registration.registration_colored_icp(
            source_aligned, target_cloud, 0.05, 
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        
        # Combine transformations
        final_transformation = reg_p2p.transformation @ initial_transformation
        
        transformations.append(final_transformation)
        print(f"Point cloud {i} registered with fitness: {reg_p2p.fitness}, RMSE: {reg_p2p.inlier_rmse}")
        
    return transformations


def estimate_transformation_from_cameras(source_cameras, target_cameras):
    """
    Estimate transformation between point clouds using camera positions
    using a simplified version of Umeyama algorithm
    """
    if len(source_cameras) < 3 or len(target_cameras) < 3:
        # Not enough camera positions for reliable estimation
        return np.eye(4)
    
    # Find corresponding cameras by proximity
    # This works if the camera positions are in sequential order
    s_cam = source_cameras
    t_cam = target_cameras
    
    # If the number of cameras differs, use the minimum length
    min_len = min(len(s_cam), len(t_cam))
    
    # Try to find best alignment using first and last few cameras
    source_pts = np.vstack([s_cam[:min(3, min_len)], s_cam[-min(3, min_len):]])
    target_pts = np.vstack([t_cam[:min(3, min_len)], t_cam[-min(3, min_len):]])
    
    # Calculate centroids
    source_centroid = np.mean(source_pts, axis=0)
    target_centroid = np.mean(target_pts, axis=0)
    
    # Center the points
    source_centered = source_pts - source_centroid
    target_centered = target_pts - target_centroid
    
    # Compute rotation using singular value decomposition
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
    
    return transformation


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge multiple point clouds from VGGT predictions')
    parser.add_argument('--prediction_dirs', type=str, nargs='+', required=True,
                        help='Directories containing prediction.npz files')
    parser.add_argument('--output', type=str, default='merged_scene.glb',
                        help='Output GLB file path')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for downsampling')
    
    args = parser.parse_args()
    
    # Load predictions
    prediction_sets = []
    for dir_path in args.prediction_dirs:
        npz_path = os.path.join(dir_path, 'predictions.npz')
        if not os.path.exists(npz_path):
            print(f"Warning: No predictions.npz found in {dir_path}")
            continue
        
        predictions = dict(np.load(npz_path, allow_pickle=True))
        prediction_sets.append(predictions)
    
    # Merge point clouds
    merged_scene = merge_point_clouds(prediction_sets, args.voxel_size)
    
    # Export merged scene
    merged_scene.export(file_obj=args.output)
    print(f"Merged scene saved to {args.output}")


if __name__ == "__main__":
    main()