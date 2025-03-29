import os
import argparse
import numpy as np
import torch
import glob
import struct
from scipy.spatial.transform import Rotation
import sys

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    model.eval()
    model = model.to(device)
    return model, device

def process_images(image_dir, model, device, conf_threshold=50.0):
    """Process images with VGGT and return predictions."""
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_names)} images")
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to camera parameters...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
   
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    print("Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    return predictions, image_names

def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        
        quaternions.append(quat)
        translations.append(t)
    
    return np.array(quaternions), np.array(translations)

def filter_and_prepare_points(predictions, conf_threshold, stride=4):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Uses stride to sample fewer points for efficiency.
    """
    # Scale confidence threshold from percentage to actual value range
    max_conf = np.max(predictions["depth_conf"])
    conf_threshold = conf_threshold / 100.0 * max_conf
    
    print(f"Confidence threshold: {conf_threshold:.4f} (scaled from {conf_threshold*100/max_conf:.2f}%)")
    
    points3D = []
    point_indices = {}  # Maps point hash to its index
    image_points2D = [[] for _ in range(len(predictions["depth"]))]
    
    # Get world points and confidences
    world_points = predictions["world_points_from_depth"]  # (num_images, H, W, 3)
    confidences = predictions["depth_conf"]  # (num_images, H, W)
    
    for img_idx in range(len(world_points)):
        h, w = world_points[img_idx].shape[:2]
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                if confidences[img_idx, y, x] > conf_threshold:
                    point3D = world_points[img_idx, y, x]
                    
                    if not np.all(np.isfinite(point3D)):
                        continue
                    
                    point_hash = hash_point(point3D, scale=100)
                    
                    # Get RGB color (gray as default)
                    rgb = np.array([200, 200, 200])
                    
                    if point_hash not in point_indices:
                        point_idx = len(points3D)
                        point_indices[point_hash] = point_idx
                        
                        point_entry = {
                            "id": point_idx,
                            "xyz": point3D,
                            "rgb": rgb,
                            "error": 1.0,
                            "track": [(img_idx, len(image_points2D[img_idx]))]
                        }
                        points3D.append(point_entry)
                    else:
                        point_idx = point_indices[point_hash]
                        points3D[point_idx]["track"].append((img_idx, len(image_points2D[img_idx])))
                    

                    image_points2D[img_idx].append((x, y, point_indices[point_hash]))
    
    print(f"Generated {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations")
    return points3D, image_points2D

def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    # Quantize point coordinates
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)

def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE"
            
            # Extract parameters for PINHOLE model: fx, fy, cx, cy
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
            
            # Write camera parameters
            f.write(f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")

def write_colmap_images_txt(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")
        
        for i in range(len(quaternions)):
            image_id = i + 1  
            camera_id = i + 1  # Assuming each image has its own camera
            
            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]

            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n")
            
            points_line = " ".join([f"{x} {y} {point3d_id+1}" for x, y, point3d_id in image_points2D[i]])
            f.write(f"{points_line}\n")

def write_colmap_points3D_txt(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        avg_track_length = sum(len(point["track"]) for point in points3D) / len(points3D) if points3D else 0
        f.write(f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n")
        
        for point in points3D:
            point_id = point["id"] + 1 
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]
            
            track = " ".join([f"{img_id+1} {point2d_idx}" for img_id, point2d_idx in point["track"]])
            
            f.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n")

def write_colmap_cameras_bin(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of cameras (uint64)
        fid.write(struct.pack('<Q', len(intrinsics)))
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model_id = 1  # PINHOLE model = 1
            
            # Extract parameters for PINHOLE model: fx, fy, cx, cy
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Model ID (uint32)
            fid.write(struct.pack('<I', model_id))
            # Width (uint64)
            fid.write(struct.pack('<Q', image_width))
            # Height (uint64)
            fid.write(struct.pack('<Q', image_height))
            
            # Parameters (double)
            fid.write(struct.pack('<dddd', fx, fy, cx, cy))

def write_colmap_images_bin(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of images (uint64)
        fid.write(struct.pack('<Q', len(quaternions)))
        
        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = i + 1
            
            qw, qx, qy, qz = quaternions[i].astype(float)
            tx, ty, tz = translations[i].astype(float)
            
            image_name = os.path.basename(image_names[i]).encode()
            points = image_points2D[i]
            
            # Image ID (uint32)
            fid.write(struct.pack('<I', image_id))
            # Quaternion (double): qw, qx, qy, qz
            fid.write(struct.pack('<dddd', qw, qx, qy, qz))
            # Translation (double): tx, ty, tz
            fid.write(struct.pack('<ddd', tx, ty, tz))
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Image name
            fid.write(struct.pack('<I', len(image_name)))
            fid.write(image_name)
            
            # Write number of 2D points (uint64)
            fid.write(struct.pack('<Q', len(points)))
            
            # Write 2D points: x, y, point3D_id
            for x, y, point3d_id in points:
                fid.write(struct.pack('<dd', float(x), float(y)))
                fid.write(struct.pack('<Q', point3d_id + 1))

def write_colmap_points3D_bin(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of points (uint64)
        fid.write(struct.pack('<Q', len(points3D)))
        
        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"].astype(float)
            r, g, b = point["rgb"].astype(np.uint8)
            error = float(point["error"])
            track = point["track"]
            
            # Point ID (uint64)
            fid.write(struct.pack('<Q', point_id))
            # Position (double): x, y, z
            fid.write(struct.pack('<ddd', x, y, z))
            # Color (uint8): r, g, b
            fid.write(struct.pack('<BBB', int(r), int(g), int(b)))
            # Error (double)
            fid.write(struct.pack('<d', error))
            
            # Track: list of (image_id, point2D_idx)
            fid.write(struct.pack('<Q', len(track)))
            for img_id, point2d_idx in track:
                fid.write(struct.pack('<II', img_id + 1, point2d_idx))

def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="colmap_output", 
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=50.0, 
                        help="Confidence threshold (0-100) for including points")
    parser.add_argument("--binary", action="store_true", 
                        help="Output binary COLMAP files instead of text")
    parser.add_argument("--stride", type=int, default=1, 
                        help="Stride for point sampling (higher = fewer points)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    model, device = load_model()
    predictions, image_names = process_images(
        args.image_dir, model, device, args.conf_threshold)
    
    print("Converting camera parameters to COLMAP format...")
    quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])

    print(f"Filtering points with confidence threshold {args.conf_threshold}% and stride {args.stride}...")
    points3D, image_points2D = filter_and_prepare_points(
        predictions, args.conf_threshold, args.stride)
    
    height, width = predictions["depth"].shape[1:3]
    
    print(f"Writing {'binary' if args.binary else 'text'} COLMAP files to {args.output_dir}...")
    if args.binary:
        write_colmap_cameras_bin(
            os.path.join(args.output_dir, "cameras.bin"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_bin(
            os.path.join(args.output_dir, "images.bin"), 
            quaternions, translations, image_points2D, image_names)
        write_colmap_points3D_bin(
            os.path.join(args.output_dir, "points3D.bin"), 
            points3D)
    else:
        write_colmap_cameras_txt(
            os.path.join(args.output_dir, "cameras.txt"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_txt(
            os.path.join(args.output_dir, "images.txt"), 
            quaternions, translations, image_points2D, image_names)
        write_colmap_points3D_txt(
            os.path.join(args.output_dir, "points3D.txt"), 
            points3D)
    
    print(f"COLMAP files successfully written to {args.output_dir}")

if __name__ == "__main__":
    main()