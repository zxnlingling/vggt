#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT.from_pretrained("facebook/VGGT-1B")  # alternative loading method

# model = VGGT()
# _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
# model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your input.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    print(f"extrinsic.shape {extrinsic.shape} \n")
    print(f"extrinsic {extrinsic}")
    print(f"extrinsic.shape {intrinsic.shape} \n")
    print(f"intrinsic {intrinsic}")
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy arrays
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    print(world_points.shape)
    print(world_points)
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    print(f"predictions are: {predictions}")
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images folder
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    Create a new folder (target_dir/images) and copy user-provided images or extract frames from video.
    Returns (target_dir, list of image paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if folder already exists (unlikely)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_path in input_images:
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        video_path = input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # extract 1 frame per second

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort images
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Reconstruction: performs the 3D reconstruction and saves the GLB file
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the images in target_dir/images.
    Returns the path to the GLB file and a log message.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please provide valid input."

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Get list of image files in target directory
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    frame_filter_choices = ["All"] + [f"{i}: {filename}" for i, filename in enumerate(all_files)]

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions to disk
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}{frame_filter.replace('.', '').replace(':', '').replace(' ', '')}"
        f"maskb{mask_black_bg}maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    log_msg = f"Reconstruction successful. Processed {len(all_files)} frames."

    return glbfile, log_msg


# -------------------------------------------------------------------------
# 4) Main function to run from command line
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VGGT 3D Reconstruction from images/video (command-line mode)"
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        default=None,
        help="Paths to one or more input image files.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a directory containing input images.",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence threshold percentage (default: 50.0).",
    )
    parser.add_argument(
        "--mask_black_bg",
        action="store_true",
        help="Filter black background points (flag; default: False).",
    )
    parser.add_argument(
        "--mask_white_bg",
        action="store_true",
        help="Filter white background points (flag; default: False).",
    )
    parser.add_argument(
        "--show_cam",
        action="store_true",
        help="Include camera visualization (flag; default: False).",
    )
    parser.add_argument(
        "--mask_sky",
        action="store_true",
        help="Filter sky points (flag; default: False).",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
        default="Depthmap and Camera Branch",
        help="Prediction mode to use.",
    )
    args = parser.parse_args()

    # Ensure that at least one type of input is provided.
    if args.video is None and args.image_dir is None and (args.images is None or len(args.images) == 0):
        parser.error("You must provide at least a video file, an image directory, or one or more image files.")

    # If an image directory is provided, use it to build a list of image files.
    image_list = None
    if args.image_dir is not None:
        image_list = []
        # Look for common image file types.
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            image_list.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
        image_list = sorted(image_list)
        if not image_list:
            parser.error(f"No image files found in directory: {args.image_dir}")
    elif args.images is not None and len(args.images) > 0:
        image_list = args.images

    # Process inputs: copy images / extract frames from video
    target_dir, image_paths = handle_uploads(args.video, image_list)
    print("Images for reconstruction:")
    for ipath in image_paths:
        print("  ", ipath)

    # For command-line, we simply use "All" for frame filtering.
    frame_filter = "All"

    # Run reconstruction
    glbfile, log_msg = gradio_demo(
        target_dir,
        conf_thres=args.conf_thres,
        frame_filter=frame_filter,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        prediction_mode=args.prediction_mode,
    )

    print(log_msg)
    print("Output GLB file saved at:", glbfile)


if __name__ == "__main__":
    main()