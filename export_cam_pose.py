#!/usr/bin/env python
import os
import numpy as np
import json
import csv
import glob
import argparse

def extract_camera_transforms(extrinsics):
    """Extract camera positions and orientations from extrinsic matrices"""
    cameras = []
    for ext in extrinsics:
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :4] = ext
        
        # Extract rotation matrix and position
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Get camera center in world coordinates (-R^T * t)
        center = -np.dot(R.T, t)
        
        # Store camera data
        camera_data = {
            'position': center.tolist(),
            'rotation_matrix': R.tolist(),
            'transform_matrix': T.tolist()
        }
        cameras.append(camera_data)
    
    return cameras

def export_cameras_csv(cameras, output_path):
    """Export camera data to CSV format"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['frame', 'pos_x', 'pos_y', 'pos_z', 
                        'r11', 'r12', 'r13',
                        'r21', 'r22', 'r23',
                        'r31', 'r32', 'r33'])
        
        # Write data for each camera
        for i, cam in enumerate(cameras):
            pos = cam['position']
            rot = cam['rotation_matrix']
            row = [i,
                  pos[0], pos[1], pos[2],
                  rot[0][0], rot[0][1], rot[0][2],
                  rot[1][0], rot[1][1], rot[1][2],
                  rot[2][0], rot[2][1], rot[2][2]]
            writer.writerow(row)

def export_cameras_json(cameras, output_path):
    """Export camera data to JSON format"""
    camera_data = {
        'cameras': [
            {
                'frame': i,
                'position': cam['position'],
                'rotation_matrix': cam['rotation_matrix'],
                'transform_matrix': cam['transform_matrix']
            }
            for i, cam in enumerate(cameras)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Export camera poses from inference results to CSV and JSON"
    )
    parser.add_argument(
        "--npz_files", type=str, required=True, 
        help="Path to directory containing NPZ files or glob pattern to match them"
    )
    parser.add_argument(
        "--output_prefix", type=str, default="camera_poses0", 
        help="Prefix for output files (will create .csv and .json files)"
    )
    
    args = parser.parse_args()
    
    # Get all NPZ files
    if os.path.isdir(args.npz_files):
        npz_paths = glob.glob(os.path.join(args.npz_files, "**", "*.npz"), recursive=True)
    else:
        npz_paths = glob.glob(args.npz_files)
    
    if not npz_paths:
        raise ValueError(f"No NPZ files found at {args.npz_files}")
    
    print(f"Found {len(npz_paths)} NPZ files")
    
    # Collect all camera transforms
    all_cameras = []
    
    for npz_path in npz_paths:
        print(f"Processing {npz_path}")
        predictions = dict(np.load(npz_path, allow_pickle=True))
        
        if "extrinsic" in predictions:
            extrinsics = predictions["extrinsic"]
            cameras = extract_camera_transforms(extrinsics)
            all_cameras.extend(cameras)
    
    if not all_cameras:
        raise ValueError("No camera transforms found in the provided NPZ files")
    
    print(f"Extracted {len(all_cameras)} camera transforms")
    
    # Export to CSV and JSON
    csv_path = f"{args.output_prefix}.csv"
    json_path = f"{args.output_prefix}.json"
    
    export_cameras_csv(all_cameras, csv_path)
    export_cameras_json(all_cameras, json_path)
    
    print(f"Exports complete!")
    print(f"CSV file saved to: {csv_path}")
    print(f"JSON file saved to: {json_path}")

if __name__ == "__main__":
    main()