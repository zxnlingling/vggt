import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_predictions(npz_file):
    """Load and return the predictions from an NPZ file."""
    if not os.path.exists(npz_file):
        print(f"Error: File {npz_file} does not exist.")
        sys.exit(1)
    
    try:
        # Load the NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        # Convert to dictionary for easier handling
        predictions = {key: data[key] for key in data.files}
        
        return predictions
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        sys.exit(1)

def print_predictions_summary(predictions):
    """Print a summary of the predictions."""
    print("\n======== PREDICTIONS SUMMARY ========")
    print(f"Number of entries: {len(predictions)}")
    
    for key, value in predictions.items():
        if isinstance(value, np.ndarray):
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Type: {value.dtype}")
            
            # Print some statistics for numerical arrays
            if np.issubdtype(value.dtype, np.number):
                try:
                    print(f"  Min: {np.min(value)}")
                    print(f"  Max: {np.max(value)}")
                    print(f"  Mean: {np.mean(value)}")
                except:
                    print("  (Could not compute statistics)")
                
            # Print a sample of the data
            print("  Sample data:")
            if value.size > 0:
                # For multi-dimensional arrays, flatten for sample display
                flat_value = value.reshape(-1)
                sample_size = min(5, flat_value.size)
                print(f"    {flat_value[:sample_size]} ...")
        else:
            print(f"\n{key}: {value}")

def print_full_array(key, array):
    """Print the full contents of an array."""
    print(f"\n======== FULL CONTENTS OF '{key}' ========")
    print(f"Shape: {array.shape}")
    print(f"Type: {array.dtype}")
    
    # For large arrays, ask for confirmation
    if array.size > 1000:
        confirm = input(f"Warning: This array has {array.size} elements. Do you want to print it all? (y/n): ")
        if confirm.lower() != 'y':
            print("Printing canceled.")
            return
    
    # Print the array
    print("\nArray contents:")
    print(array)

def visualize_depth_maps(predictions):
    """Visualize the depth maps from the predictions."""
    if 'depth' not in predictions:
        print("No depth maps found in predictions.")
        return
    
    depth_maps = predictions['depth']
    
    # Determine how many depth maps to display
    num_frames = depth_maps.shape[0] if depth_maps.ndim > 3 else 1
    
    # For multi-dimensional arrays, adapt accordingly
    if depth_maps.ndim == 4:  # (frames, height, width, 1)
        num_frames = min(4, depth_maps.shape[0])  # Show up to 4 frames
        fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
        
        if num_frames == 1:
            axes = [axes]  # Make it iterable for consistency
            
        for i, ax in enumerate(axes):
            if i < depth_maps.shape[0]:
                # Remove the channel dimension if it exists
                if depth_maps.shape[-1] == 1:
                    depth_map = depth_maps[i, :, :, 0]
                else:
                    depth_map = depth_maps[i]
                
                im = ax.imshow(depth_map, cmap='viridis')
                ax.set_title(f"Frame {i}")
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

def visualize_point_cloud(predictions):
    """Visualize the 3D point cloud."""
    point_cloud_key = None
    if 'world_points_from_depth' in predictions:
        point_cloud_key = 'world_points_from_depth'
    
    if point_cloud_key is None:
        print("No point cloud data found in predictions.")
        return
    
    points = predictions[point_cloud_key]
    
    # Choose a specific frame if multiple are available
    if points.ndim == 4:  # (frames, height, width, 3)
        frame_idx = 0
        print(f"Showing point cloud for frame {frame_idx} (shape: {points.shape})")
        points = points[frame_idx]
    
    # Flatten the spatial dimensions to get a list of 3D points
    points_flat = points.reshape(-1, 3)
    
    # Create a scatter plot in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample if there are too many points
    max_points = 10000
    if len(points_flat) > max_points:
        # Random sampling
        idx = np.random.choice(len(points_flat), max_points, replace=False)
        points_flat = points_flat[idx]
    
    # Plot the points
    ax.scatter(
        points_flat[:, 0], 
        points_flat[:, 1], 
        points_flat[:, 2], 
        c='b', 
        marker='.', 
        alpha=0.5,
        s=1
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Read and analyze predictions.npz file from VGGT model.')
    parser.add_argument('npz_file', help='Path to the predictions.npz file')
    parser.add_argument('--key', help='Specific key to print from the npz file')
    parser.add_argument('--visualize', choices=['depth', 'points', 'all'], 
                        help='Visualize specific data: depth maps or 3D points')
    
    args = parser.parse_args()
    
    # Load predictions
    predictions = load_predictions(args.npz_file)
    
    # If a specific key is provided, print it and exit
    if args.key:
        if args.key in predictions:
            print_full_array(args.key, predictions[args.key])
            return
        else:
            print(f"Error: Key '{args.key}' not found.")
            print("Available keys:")
            for key in predictions.keys():
                print(f"- {key}")
            return
    
    # Print summary
    print_predictions_summary(predictions)
    
    # Handle visualization request
    if args.visualize:
        if args.visualize == 'depth' or args.visualize == 'all':
            visualize_depth_maps(predictions)
        if args.visualize == 'points' or args.visualize == 'all':
            visualize_point_cloud(predictions)
    else:
        # Interactive mode
        while True:
            print("\n======== OPTIONS ========")
            print("1: List all available keys")
            print("2: Print contents of a specific key")
            print("3: Visualize depth maps")
            print("4: Visualize 3D point cloud")
            print("5: Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                print("\nAvailable keys:")
                for key in sorted(predictions.keys()):
                    print(f"- {key}")
            elif choice == '2':
                key = input("Enter the key name: ")
                if key in predictions:
                    print_full_array(key, predictions[key])
                else:
                    print(f"Error: Key '{key}' not found.")
                    print("Available keys:")
                    for k in sorted(predictions.keys()):
                        print(f"- {k}")
            elif choice == '3':
                visualize_depth_maps(predictions)
            elif choice == '4':
                visualize_point_cloud(predictions)
            elif choice == '5':
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()