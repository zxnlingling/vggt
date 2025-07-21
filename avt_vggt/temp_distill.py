# model: timm
# backbone: ViT-B-16
# dataset: xxx # objaverse
# matcher: vggt

# hydra:
#   run:
#     dir: outputs/${model}/${matcher}/${backbone}/${dataset}

#   sweep:
#     dir: checkpoints
#     subdir: ${backbone}

# evaluation_methods:
#   - semantic_transfer
#   - tracking
#   # - pose





import os
import gc
import cv2
import sys
import copy
import glob
import json
import math
import timm
import hydra
import torch
import types
import random
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
# import albumentations as A
from einops import rearrange
from datetime import datetime
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.parameter as Parameter
from typing import Any, Dict, Mapping
import torch.nn.modules.utils as nn_utils
from torch.utils.data import ConcatDataset
from utils.vggt_utils import get_model_para
from torchvision.transforms import functional
import torchvision.transforms.functional as VF
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

# --- VGGT Imports ---
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
# ---------------------

# ----------------
#     Pipeline
# ----------------
# ref stands for refactoring
# from data_utils.dataset_mast3r_objaverse import AugmentedCustomObjaverseDataset
# from data_utils.dataset_mast3r_scannetpp import AugmentedCustomScanNetPPDataset
# from data_utils.dataset_vggt_objaverse import ObjaverseVGGTDataset
# from data_utils.dataset_vggt_scannetpp import ScanNetPPVGGTDataset

EPS = 1e-08
imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def _fix_pos_enc(patch_size, stride_hw):
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        # compute number of tokens taking stride into account
        w0 = 1 + (w - patch_size) // stride_hw[1]
        h0 = 1 + (h - patch_size) // stride_hw[0]
        assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                        stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return interpolate_pos_encoding

def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance

def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=False, normalize_w=False, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = (samples[:, :, :, :, 2]) / (t - 1)  # normalize to [0,1]
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values

# copied from OmniMotion
def gen_grid(h_start, w_start, h_end, w_end, step_h, step_w, device, normalize=False, homogeneous=False):
    """Generate a grid of coordinates in the image frame.
    Args:
        h, w: height and width of the grid.
        device: device to put the grid on.
        normalize: whether to normalize the grid coordinates to [-1, 1].
        homogeneous: whether to return the homogeneous coordinates. homogeneous coordinates are 3D coordinates.
    Returns:"""
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h_end, device=device)
        lin_x = torch.linspace(-1., 1., steps=w_end, device=device)
    else:
        lin_y = torch.arange(h_start, h_end, step=step_h, device=device)
        lin_x = torch.arange(w_start, w_end, step=step_w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]

class TrackerHead(nn.Module):
    def __init__(self,
                 in_channels=1,
                 hidden_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 
                 patch_size=14,
                 step_h=14,
                 step_w=14,
                 argmax_radius=35,
                 video_h=480,
                 video_w=640):
        super(TrackerHead, self).__init__()
        
        padding = kernel_size // 2
        
        self.softmax = nn.Softmax(dim=2)
        self.argmax_radius = argmax_radius
        self.patch_size = patch_size
        self.step_h = step_h
        self.step_w = step_w
        self.video_h=video_h
        self.video_w=video_w
    
    def soft_argmax(self, heatmap, argmax_indices):
        """
        heatmap: shape (B, H, W)
        """
        h_start = self.patch_size // 2
        w_start = self.patch_size // 2
        h_end = ((self.video_h - 2 * h_start) // self.step_h) * self.step_h + h_start + math.ceil(self.step_h / 2)
        w_end = ((self.video_w - 2 * w_start) // self.step_w) * self.step_w + w_start + math.ceil(self.step_w / 2)
        grid = gen_grid(h_start=h_start, w_start=w_start, h_end=h_end, w_end=w_end, step_h=self.step_h, step_w=self.step_w,
                        device=heatmap.device, normalize=False, homogeneous=False) # shape (H, W, 2)
        grid = grid.unsqueeze(0).repeat(heatmap.shape[0], 1, 1, 1) # stack and repeat grid to match heatmap shape (B, H, W, 2)
        
        row, col = argmax_indices
        argmax_coord = torch.stack((col*self.step_w+w_start, row*self.step_h+h_start), dim=-1) # (x,y) coordinates, shape (B, 2)
        
        # generate a mask of a circle of radius radius around the argmax_coord (B, 2) in heatmap (B, H, W, 2)
        mask = torch.norm((grid - argmax_coord.unsqueeze(1).unsqueeze(2)).to(torch.float32), dim=-1) <= self.argmax_radius # shape (B, H, W)
        heatmap = heatmap * mask
        hm_sum = torch.sum(heatmap, dim=(1, 2)) # B
        hm_zero_indices = hm_sum < 1e-8
        
        # for numerical stability
        if sum(hm_zero_indices) > 0:
            uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
            heatmap[hm_zero_indices] += uniform_w[:, None, None]
            heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
            hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        point = torch.sum(grid * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (B, 2)

        return point
    
    def softmax_heatmap(self, hm):
        b, c, h, w = hm.shape
        hm_sm = rearrange(hm, "b c h w -> b c (h w)") # shape (B, 1, H*W)
        hm_sm = self.softmax(hm_sm) # shape (B, 1, H*W)
        hm_sm = rearrange(hm_sm, "b c (h w) -> b c h w", h=h, w=w) # shape (B, 1, H, W)
        return hm_sm
    
    def forward(self, cost_volume):
        """
        cost_volume: shape (B, C, H, W)
        """
        
        range_normalizer = RangeNormalizer(shapes=(self.video_w, self.video_h)) # shapes are (W, H), correpsonding to (x, y) coordinates
        
        # crop heatmap around argmax point
        argmax_flat = torch.argmax(rearrange(cost_volume[:, 0], "b h w -> b (h w)"), dim=1)
        argmax_indices = (argmax_flat // cost_volume[:, 0].shape[-1], argmax_flat % cost_volume[:, 0].shape[-1])

        refined_heatmap = self.softmax_heatmap(cost_volume) # shape (B, 1, H, W)
        point = self.soft_argmax(refined_heatmap.squeeze(1),
                                 argmax_indices) # shape (B, 2), (x,y) coordinates
        return range_normalizer(point, dst=(-1,1), dims=[0, 1]) # shape (B, 2)
    
class Tracker(nn.Module):
    def __init__(
        self,
        dino_features,
        video=None,
        ckpt_path="",
        dino_patch_size=14,
        stride=7,
        device="cuda:0",
        ):
        super().__init__()

        self.stride = stride
        self.dino_patch_size = dino_patch_size
        self.device = device
        self.refined_features = None
        self.ckpt_path = ckpt_path
        
        self.video = video
        
        # DINO embed
        self.dino_embed_video = dino_features  # T x C x H x W

        # CNN-Refiner
        t, c, h, w = self.video.shape
        self.cmap_relu = nn.ReLU(inplace=True)
        self.tracker_head = TrackerHead(patch_size=dino_patch_size,
                                        step_h=stride,
                                        step_w=stride,
                                        video_h=h,
                                        video_w=w).to(device)
        self.range_normalizer = RangeNormalizer(shapes=(w, h, self.video.shape[0]))
  
    def get_dino_embed_video(self, frames_set_t):
        dino_emb = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)] if frames_set_t.device != self.dino_embed_video.device else self.dino_embed_video[frames_set_t]
        return dino_emb
    
    def normalize_points_for_sampling(self, points):
        t, c, vid_h, vid_w = self.video.shape
        h = vid_h
        w = vid_w
        patch_size = self.dino_patch_size
        stride = self.stride
        
        last_coord_h =( (h - patch_size) // stride ) * stride + (patch_size / 2)
        last_coord_w =( (w - patch_size) // stride ) * stride + (patch_size / 2)
        ah = 2 / (last_coord_h - (patch_size / 2))
        aw = 2 / (last_coord_w - (patch_size / 2))
        bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
        bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
        
        a = torch.tensor([[aw, ah, 1]]).to(self.device)
        b = torch.tensor([[bw, bh, 0]]).to(self.device)
        normalized_points = a * points + b
        
        # normalized_points = points.clone()
        # h = vid_h // patch_size * patch_size
        # w = vid_w // patch_size * patch_size

        # # convert keypoint location to pixel center
        # normalized_points[..., 0] = ((normalized_points[..., 0] + 0.5) / w) * 2 - 1  # x coordinates
        # normalized_points[..., 1] = ((normalized_points[..., 1] + 0.5) / h) * 2 - 1  # y coordinates
        
        return normalized_points
    
    def sample_embeddings(self, embeddings, source_points):
        """embeddings: T x C x H x W. source_points: B x 3, where the last dimension is (x, y, t), x and y are in [-1, 1]"""
        t, c, h, w = embeddings.shape
        sampled_embeddings = bilinear_interpolate_video(video=rearrange(embeddings, "t c h w -> 1 c t h w"),
                                                               points=source_points,
                                                               h=h,
                                                               w=w,
                                                               t=t,
                                                               normalize_w=False,
                                                               normalize_h=False,
                                                               normalize_t=True)
        sampled_embeddings = sampled_embeddings.squeeze()
        if len(sampled_embeddings.shape) == 1:
            sampled_embeddings = sampled_embeddings.unsqueeze(1)
        sampled_embeddings = sampled_embeddings.permute(1,0)
        return sampled_embeddings
    
    def uncache_refined_embeddings(self, move_dino_to_gpu=False):
        self.refined_features = None
        torch.cuda.empty_cache()
        gc.collect()
        if move_dino_to_gpu:
            self.dino_embed_video = self.dino_embed_video.to("cuda")
    
    def get_corr_maps_for_frame_set(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps_set = torch.einsum("bc,nchw->bnhw", source_embeddings, frame_embeddings_set)
        corr_maps = corr_maps_set[torch.arange(source_embeddings.shape[0]), target_frame_indices.int(), :, :]
        
        embeddings_norm = frame_embeddings_set.norm(dim=1)
        target_embeddings_norm = embeddings_norm[target_frame_indices.int()]
        source_embeddings_norm = source_embeddings.norm(dim=1).unsqueeze(-1).unsqueeze(-1)
        corr_maps_norm = (source_embeddings_norm * target_embeddings_norm)
        corr_maps = corr_maps / torch.clamp(corr_maps_norm, min=EPS)
        corr_maps = rearrange(corr_maps, "b h w -> b 1 h w")
        
        return corr_maps
    
    def get_point_predictions_from_embeddings(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps = self.get_corr_maps_for_frame_set(source_embeddings, frame_embeddings_set, target_frame_indices)
        coords = self.tracker_head(self.cmap_relu(corr_maps))
        return coords
    
    def get_point_predictions(self, inp, frame_embeddings):
        source_points_unnormalized, source_frame_indices, target_frame_indices, _ = inp
        source_points = self.normalize_points_for_sampling(source_points_unnormalized)
        # print(frame_embeddings.device, source_points.device, source_frame_indices.device, target_frame_indices.device)
        source_embeddings = self.sample_embeddings(frame_embeddings, torch.cat([ source_points[:, :-1], source_frame_indices[:, None] ], dim=1)) # B x C
        return self.get_point_predictions_from_embeddings(source_embeddings, frame_embeddings, target_frame_indices)

    def forward(self, inp, use_raw_features=False):
        """
        inp: source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t; where
        source_points_unnormalized: B x 3. ((x, y, t) in image scale - NOT normalized)
        source_frame_indices: the indices of frames of source points in frames_set_t
        target_frame_indices: the indices of target frames in frames_set_t
        frames_set_t: N, 0 to T-1 (NOT normalized)
        """
        frames_set_t = inp[-1]
        
        if use_raw_features:
            frame_embeddings = raw_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        self.frame_embeddings = frame_embeddings
        self.raw_embeddings = raw_embeddings
        coords = self.get_point_predictions(inp, frame_embeddings)

        return coords

class RangeNormalizer(torch.nn.Module):
    """
    Scales dimensions to specific ranges.
    Will be used to normalize pixel coords. & time to destination ranges.
    For example: [0, H-1] x [0, W-1] x [0, T-1] -> [0,1] x [0,1] x [0,1]

    Args:
         shapes (tuple): represents the "boundaries"/maximal values for each input dimension.
            We assume that the dimensions range from 0 to max_value (as in pixels & frames).
    """
    def __init__(self, shapes: tuple, device='cuda'):
        super().__init__()

        normalizer = torch.tensor(shapes).float().to(device) - 1
        self.register_buffer("normalizer", normalizer)

    def forward(self, x, dst=(0, 1), dims=[0, 1, 2]):
        """
        Normalizes input to specific ranges.
        
            Args:       
                x (torch.tensor): input data
                dst (tuple, optional): range inputs where normalized to. Defaults to (0, 1).
                dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].
                
            Returns:
                normalized_x (torch.tensor): normalized input data
        """
        normalized_x = x.clone()
        normalized_x[:, dims] = x[:, dims] / self.normalizer[dims] # normalize to [0,1]
        normalized_x[:, dims] = (dst[1] - dst[0]) * normalized_x[:, dims] + dst[0] # shift range to dst

        return normalized_x
    
    def unnormalize(self, normalized_x:torch.tensor, src=(0, 1), dims=[0, 1, 2]):
        """Runs to reverse process of forward, unnormalizes input to original scale.

        Args:
            normalized_x (torch.tensor): input data
            src (tuple, optional): range inputs where normalized to. Defaults to (0, 1). unnormalizes from src to original scales.
            dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].

        Returns:
            x (torch.tensor): unnormalized input data
        """
        x = normalized_x.clone()
        x[:, dims] = (normalized_x[:, dims] - src[0]) / (src[1] - src[0]) # shift range to [0,1]
        x[:, dims] = x[:, dims] * self.normalizer[dims] # unnormalize to original ranges
        return x
    
# ---- Functions for generating trajectories ----
def generate_trajectory_input(query_point, video, start_t=None, end_t=None):
    """
    Receives a single point (x,y,t) and the video, and generates input for Tracker model.
    Args:
        query_point: shape 3. (x,y,t).
        video: shape T x H x W x 3.
    Returns:
        source_points, source_frame_indices, target_frame_indices, frames_set_t.
        source_points: query_point repeated rest times. shape rest x 3. (x,y,t).
        source_frame_indices: [0] repeated rest times. shape rest x 1.
        target_frame_indices: 0 to rest-1. shape rest.
        frames_set_t: [query_point[0, 2], start_t, ..., end_t]. shape rest + 1.
    """
    start_t = 0 if start_t is None else start_t
    end_t = video.shape[0] if end_t is None else end_t
    video_subset = video[start_t:end_t]
    rest = video_subset.shape[0]
    device = video.device
    
    source_points = query_point.unsqueeze(0).repeat(rest, 1) # rest x 3

    frames_set_t = torch.arange(start_t, end_t, dtype=torch.long, device=device) # rest
    frames_set_t = torch.cat([ torch.tensor([query_point[2]], device=device), frames_set_t ]).int() # rest + 1
    source_frame_indices = torch.tensor([0], device=device).repeat(end_t-start_t) # rest
    target_frame_indices = torch.arange(rest, dtype=torch.long, device=device) + 1 # T
    
    return source_points, source_frame_indices, target_frame_indices, frames_set_t

@torch.no_grad()
def generate_trajectory(query_point:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                               batch_size=None) -> torch.tensor:
    """
    Genrates trajectory using tracker predictions for all timesteps.
    Returns:
        trajectory_pred: rest x 3. (x,y,t) coordinates for each timestep.
    """
    batch_size = video.shape[0] if batch_size is None else batch_size
    
    trajectory_pred = []
    for start_t in range(0, video.shape[0], batch_size):
        end_t = min(start_t + batch_size, video.shape[0])
        trajectory_input = generate_trajectory_input(query_point, video, start_t=start_t, end_t=end_t)
        trajectory_coordinate_preds_normalized = model(trajectory_input, use_raw_features=use_raw_features)
        trajectory_coordinate_preds = range_normalizer.unnormalize(trajectory_coordinate_preds_normalized, dims=[0,1], src=dst_range)
        trajectory_timesteps = trajectory_input[-1][1:].to(dtype=torch.float32) # rest
        trajectory_pred_cur = torch.cat([trajectory_coordinate_preds, trajectory_timesteps.unsqueeze(dim=1)], dim=1)
        trajectory_pred.append(trajectory_pred_cur)
    trajectory_pred = torch.cat(trajectory_pred, dim=0)
    return trajectory_pred
    
@torch.no_grad()
def generate_trajectories(query_points:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                                 batch_size=None) -> torch.tensor:
    """
    Genrates trajectories using tracker predictions. wraps generate_trajectory function.
    Returns:
        trajectories: len(query_points) x rest x 3. (x,y,t) coordinates for each trajectory.
    """
    trajectories_list = []
    query_points = query_points.to(dtype=torch.float32) # just in case
    for query_point in query_points:
        trajectory_pred = generate_trajectory(query_point=query_point, video=video, model=model, range_normalizer=range_normalizer, dst_range=dst_range, use_raw_features=use_raw_features,
                                                     batch_size=batch_size)
        trajectories_list.append(trajectory_pred)
    trajectories = torch.stack(trajectories_list)
    return trajectories

class ModelInference(torch.nn.Module):
    def __init__(
        self,
        model: Tracker,
        range_normalizer: RangeNormalizer,
        anchor_cosine_similarity_threshold: float = 0.5,
        cosine_similarity_threshold: float = 0.5,
        ) -> None:
        super().__init__()


        self.model = model
        self.model.eval()

        self.range_normalizer = range_normalizer
        self.anchor_cosine_similarity_threshold = anchor_cosine_similarity_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
    
    def compute_trajectories(self, query_points: torch.Tensor, batch_size=None,) -> torch.Tensor:
        trajecroies = generate_trajectories(
            query_points=query_points,
            model=self.model,
            video=self.model.video,
            range_normalizer=self.range_normalizer,
            dst_range=(-1,1),
            use_raw_features=True,
            batch_size=batch_size,
        )
        return trajecroies
    
    # ----------------- Cosine Similarity -----------------
    def compute_trajectory_cos_sims(self, trajectories, query_points) -> torch.Tensor:
        """Compute cosine similarities between trajectories and query points.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. (x, y, t).
            query_points (torch.Tensor): Query points. N x 3. used for retrieving corresponding query frames.
        Returns:
            trajectories_cosine_similarities (torch.Tensor): Cosine similarities between trajectories and query points. N x T."""
        # compute refined_features_at_trajectories
        N, T = trajectories.shape[:2]
        trajectories_normalized = self.model.normalize_points_for_sampling(trajectories) # N x T x 3
        features = self.model.get_dino_embed_video(frames_set_t=torch.arange(0, self.model.video.shape[0]))
        refined_features_at_trajectories = self.model.sample_embeddings(features, trajectories_normalized.view(-1, 3)) # (N*T) x C
        refined_features_at_trajectories = refined_features_at_trajectories.view(N, T, -1) # N x T x C
        
        query_frames = query_points[:, 2].long() # N
        refined_features_at_query_frames = refined_features_at_trajectories[torch.arange(N).to(self.model.device), query_frames] # N x C
        trajectories_cosine_similarities = torch.nn.functional.cosine_similarity(refined_features_at_query_frames.unsqueeze(1), refined_features_at_trajectories, dim=-1) # N x T
        return trajectories_cosine_similarities


    # ----------------- Anchor Trajectories -----------------
    def _get_model_preds_at_anchors(self, model, range_normalizer, preds, anchor_indices, batch_size=None):
        """ preds: N"""
        batch_size = batch_size if batch_size is not None else preds.shape[0]
        
        cycle_coords = []
        for vis_frame in anchor_indices:
            # iterate over frames_set_t in batches of size batch_size
            coords = []
            for i in range(0, preds.shape[0], batch_size):
                end_idx = min(i + batch_size, preds.shape[0])
                frames_set_t = torch.arange(i, end_idx, device=model.device)
                frames_set_t = torch.cat([ torch.tensor([vis_frame], device=model.device), frames_set_t ]).int()
                source_frame_indices = torch.arange(1, frames_set_t.shape[0], device=model.device)
                target_frame_indices = torch.tensor([0]*(frames_set_t.shape[0]-1), device=model.device)
                inp = preds[i:end_idx], source_frame_indices, target_frame_indices, frames_set_t
                batch_coords = model(inp, use_raw_features=True) # batch_size x 3
                batch_coords = range_normalizer.unnormalize(batch_coords, src=(-1, 1), dims=[0, 1])
                coords.append(batch_coords)
            coords = torch.cat(coords)
            
            cycle_coords.append(coords[:, :2]) # prediction of a target point to the top percentile
            
        cycle_coords = torch.stack(cycle_coords) # N_anchors x T x 2
        
        return cycle_coords
    
    def compute_anchor_trajectories(self, trajectories: torch.Tensor, cos_sims: torch.Tensor, batch_size=None) -> torch.Tensor:
        N, T = trajectories.shape[:2]
        eql_anchor_cyc_predictions = {}
            
        for qp_idx in tqdm(range(N), desc=f"Interating over query points"):
            preds = trajectories[qp_idx] # (T x 3)
            anchor_frames = torch.arange(T).to(self.model.device)[cos_sims[qp_idx] >= self.anchor_cosine_similarity_threshold] # T
            cycle_coords_eql_anchor = self._get_model_preds_at_anchors(self.model, self.range_normalizer, preds=preds, anchor_indices=anchor_frames, batch_size=batch_size)
            eql_anchor_cyc_predictions[qp_idx] = cycle_coords_eql_anchor
        return eql_anchor_cyc_predictions
    
    
    # ----------------- Occlusion -----------------
    def compute_occ_pred_for_qp(self, green_trajectories_qp: torch.tensor, source_trajectories_qp: torch.tensor, traj_cos_sim_qp: torch.tensor, anch_sim_th: float, cos_sim_th: float):
        visible_at_st_frame_qp = traj_cos_sim_qp >= anch_sim_th
        dists_from_source = torch.norm(green_trajectories_qp - source_trajectories_qp[visible_at_st_frame_qp, :].unsqueeze(1), dim=-1)  # dists_from_source (M x T), dists_from_source[anchor_t, source_t] = dist

        anchor_median_errors = torch.median(dists_from_source[:, visible_at_st_frame_qp], dim=0).values  # T_vis
        median_anchor_dist_th = anchor_median_errors.max()  # float
        dists_from_source_anchor_vis = dists_from_source  # (T_vis x T)
        median_dists_from_source_anchor_vis = torch.median(dists_from_source_anchor_vis, dim=0).values  # T
        return ((median_dists_from_source_anchor_vis > median_anchor_dist_th) | (traj_cos_sim_qp < cos_sim_th))

    def compute_occlusion(self, trajectories: torch.Tensor, trajs_cos_sims: torch.Tensor, anchor_trajectories: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute occlusion for trajectories.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. trajectory for qp_idx in query_points (N).
            trajs_cos_sims (torch.Tensor): Cosine similarities between trajectories and query points. N x T. traj_cos_sims[qp_idx, t] = cos_sim
            anchor_trajectories dict(torch.Tensor): Anchor trajectories. {qp_idx: T x T x 2}. N is the number of trajectories.
        Returns:
            occ_preds (torch.Tensor): Occlusion predictions. N x T. occ_preds[qp_idx, t] = 1 if occluded, 0 otherwise.
        """

        N = trajectories.shape[0]
        occ_preds_by_dist_th_anchor_frame_vis = []

        for qp_idx in range(N):
            source_trajectories_qp = trajectories[qp_idx, :, :2] # source_trajectories_qp (T x 2)
            traj_cos_sim_qp = trajs_cos_sims[qp_idx] # cos_sim_qp (T)
            green_trajectories_qp = anchor_trajectories[qp_idx] # (T x T x 2), green_trajectories_qp[achor_t, source_t] = [x, y], source_t = start_frame
            occ_preds_by_dist_th_anchor_frame_vis.append(self.compute_occ_pred_for_qp(green_trajectories_qp, source_trajectories_qp, traj_cos_sim_qp, self.anchor_cosine_similarity_threshold, self.cosine_similarity_threshold))

        occ_preds = torch.stack(occ_preds_by_dist_th_anchor_frame_vis) # (N x T)

        return occ_preds
    
    # ----------------- Inference -----------------
    @torch.no_grad()
    def infer(self, query_points: torch.Tensor, batch_size=None, output_occ=True) -> torch.Tensor:
        """Infer trajectory and occlusion for query points.
        Args:
            query_points (torch.Tensor): Query points. N x 3. N is the number of query points. (x, y, t).
            batch_size (int): Batch size for inference. if None, all frames are inferred at once.
        Returns:
            trajectories (torch.Tensor): Predicted trajectory. N x T x 2. T is the number of time steps.
            occlusion (torch.Tensor): Predicted occlusion. N x T. T is the number of time steps."""
        trajs = self.compute_trajectories(query_points, batch_size) # N x T x 3
        if output_occ:
            cos_sims = self.compute_trajectory_cos_sims(trajs, query_points)
            anchor_trajs = self.compute_anchor_trajectories(trajs, cos_sims, batch_size)
            occ = self.compute_occlusion(trajs, cos_sims, anchor_trajs)
        else:
            occ = torch.zeros(trajs.shape[0], trajs.shape[1], device=trajs.device)
        return trajs[..., :2], occ # N x T x 2, N x T

def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale

def oneposepp(module, num_objs=None):
    stride = 16
    patch_size = 16

    # model = module.dinov2
    model = module

    root = 'data/lowtexture_test_data'
    sfm_dir = 'data/sfm_output/outputs_softmax_loftr_loftr'
    all_obj = [name for name in os.listdir(
        root) if os.path.isdir(os.path.join(root, name))]

    if num_objs is not None:
        all_obj = all_obj[:num_objs]

    threshold_1 = []
    threshold_3 = []
    threshold_5 = []

    for obj_name in all_obj:
        print(obj_name)
        anno_3d = np.load(f'{sfm_dir}/{obj_name}/anno/anno_3d_average.npz')
        keypoints3d = anno_3d['keypoints3d']

        templates = []
        all_json_fns = list(
            (Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'anno_loftr').glob('*.json'))
        for json_fn in tqdm(all_json_fns):
            idx = json_fn.stem
            anno = json.load(open(json_fn))
            keypoints2d = np.array(anno['keypoints2d'])
            assign_matrix = np.array(anno['assign_matrix'])
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-1'.format(
                obj_name.split('-')[1]) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            intrinsic = np.loadtxt(str(Path(
                root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'intrin_ba' / f'{idx}.txt'))
            # pose = np.loadtxt(str(Path(root) / obj_name / '{}-1'.format(obj_name.split('-')[1]) / 'poses_ba' / f'{idx}.txt'))

            keypoints2d = keypoints2d[assign_matrix[0]]
            kp3ds_canon = keypoints3d[assign_matrix[1]]

            rgb_resized = cv2.resize(
                rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))

            # desc = model.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).cuda().float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
            desc = model.model.forward_features(imagenet_norm(torch.from_numpy(
                rgb_resized).cuda().float().permute(2, 0, 1)[None]))[:, 1:]
            desc = desc.reshape(
                1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)

            # Only apply refine_conv if it exists in the module
            if hasattr(module, 'refine_conv'):
                desc = module.refine_conv(desc)
                
            desc_temp = interpolate_features(desc, torch.from_numpy(keypoints2d).float().cuda()[None] / 8 * patch_size,
                                             h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]

            desc_temp /= (desc_temp.norm(dim=-1, keepdim=True) + 1e-9)
            kp_temp, kp3d_temp = keypoints2d, kp3ds_canon

            templates.append((kp_temp, desc_temp, kp3d_temp))

        all_descs_temp = torch.cat([t[1] for t in templates], 0).cuda()[::1]
        all_pts3d_temp = np.concatenate([t[2] for t in templates], 0)[::1]
        # print(all_descs_temp.shape, all_pts3d_temp.shape)

        # subsample if too many
        if len(all_descs_temp) > 120000:
            idx = np.random.choice(len(all_descs_temp), 120000, replace=False)
            all_descs_temp = all_descs_temp[idx]
            all_pts3d_temp = all_pts3d_temp[idx]

        R_errs = []
        t_errs = []
        pts3d_scale = 1000
        grid_stride = 4
        test_seq = '2'

        all_img_fns = list(sorted((Path(root) / obj_name / '{}-{}'.format(
            obj_name.split('-')[1], test_seq) / 'color').glob('*.png')))[::10]
        for i, img_fn in enumerate(tqdm(all_img_fns)):
            idx = img_fn.stem
            rgb = cv2.imread(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'color' / f'{idx}.png'))[..., ::-1].copy()
            # mask = remove(rgb, only_mask=True) > 0
            intrinsic = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'intrin_ba' / f'{idx}.txt'))
            pose_gt = np.loadtxt(str(Path(root) / obj_name / '{}-{}'.format(
                obj_name.split('-')[1], test_seq) / 'poses_ba' / f'{idx}.txt'))

            with torch.no_grad():
                if i == 0:
                    x_coords = np.arange(0, rgb.shape[1], grid_stride)
                    y_coords = np.arange(0, rgb.shape[0], grid_stride)

                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
                    kp = np.column_stack(
                        (x_mesh.ravel(), y_mesh.ravel())).astype(float)

                rgb_resized = cv2.resize(
                    rgb / 255., (rgb.shape[1] // 8 * patch_size, rgb.shape[0] // 8 * patch_size))

                # desc = model.forward_features(imagenet_norm(torch.from_numpy(rgb_resized).cuda().float().permute(2, 0, 1)[None]))['x_norm_patchtokens']
                desc = model.model.forward_features(imagenet_norm(torch.from_numpy(
                    rgb_resized).cuda().float().permute(2, 0, 1)[None]))[:, 1:]
                desc = desc.reshape(
                    1, rgb.shape[0] // 8, rgb.shape[1] // 8, -1).permute(0, 3, 1, 2)

                # Only apply refine_conv if it exists in the module
                if hasattr(module, 'refine_conv'):
                    desc = module.refine_conv(desc)
                    
                desc = interpolate_features(desc, torch.from_numpy(kp).float().cuda()[None] / 8 * patch_size,
                                            h=rgb_resized.shape[0], w=rgb_resized.shape[1], normalize=False, patch_size=patch_size, stride=stride).permute(0, 2, 1)[0]
                desc /= (desc.norm(dim=-1, keepdim=True) + 1e-9)

            with torch.no_grad():
                nbr1 = []
                for d in torch.split(desc, (25000 * 10000 - 1) // all_descs_temp.shape[0] + 1):
                    sim = d @ all_descs_temp.T
                    nbr1.append(sim.argmax(-1))
                nbr1 = torch.cat(nbr1, 0)

                nbr2 = []
                for d in torch.split(all_descs_temp, (25000 * 10000 - 1) // desc.shape[0] + 1):
                    sim = d @ desc.T
                    nbr2.append(sim.argmax(-1))
                nbr2 = torch.cat(nbr2, 0)

            m_mask = nbr2[nbr1] == torch.arange(len(nbr1)).to(nbr1.device)
            # m_mask = m_mask.cpu().numpy()
            # nbr1 = nbr1.cpu().numpy()
            # nbr2 = nbr2.cpu().numpy()

            src_pts = kp[m_mask.cpu().numpy()].reshape(-1, 1, 2)
            dst_3dpts = all_pts3d_temp[nbr1[m_mask].cpu().numpy()]

            pose_pred = np.eye(4)
            if m_mask.sum() >= 4:
                _, R_exp, trans, pnp_inlier = cv2.solvePnPRansac(dst_3dpts * pts3d_scale,
                                                                 src_pts[:, 0],
                                                                 intrinsic,
                                                                 None,
                                                                 reprojectionError=8.0,
                                                                 iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP)
                trans /= pts3d_scale
                if pnp_inlier is not None:
                    if len(pnp_inlier) > 5:
                        R, _ = cv2.Rodrigues(R_exp)
                        r_t = np.concatenate([R, trans], axis=-1)
                        pose_pred = np.concatenate(
                            (r_t, [[0, 0, 0, 1]]), axis=0)

            R_err, t_err = query_pose_error(pose_pred, pose_gt)
            R_errs.append(R_err)
            t_errs.append(t_err)
            # print(R_err, t_err, cnt, len(matches), len(templates[0][0]))
        print(f'object: {obj_name}')
        for pose_threshold in [1, 3, 5]:
            acc = np.mean(
                (np.array(R_errs) < pose_threshold) & (
                    np.array(t_errs) < pose_threshold)
            )
            print(f'pose_threshold: {pose_threshold}, acc: {acc}')

            if pose_threshold == 1:
                threshold_1.append(acc)
            elif pose_threshold == 3:
                threshold_3.append(acc)
            else:
                threshold_5.append(acc)

    result = {}
    result['threshold_1'] = threshold_1
    result['threshold_3'] = threshold_3
    result['threshold_5'] = threshold_5

    metrics_df = pd.DataFrame(result)
    metrics_df['objs'] = all_obj
    metrics_df.set_index(['objs'], inplace=True)

    return metrics_df

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.
     get_trackwise_metrics: if True, the metrics will be computed for every
       track (rather than every video, which is the default).  This means
       every output tensor will have an extra axis [batch, num_tracks] rather
       than simply (batch).

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  summing_axis = (2,) if get_trackwise_metrics else (1, 2)

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=summing_axis,
  ) / np.sum(evaluation_points, axis=summing_axis)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=summing_axis,
    )
    count_visible_points = np.sum(
        visible & evaluation_points, axis=summing_axis
    )
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=summing_axis
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(
        false_positives & evaluation_points, axis=summing_axis
    )
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics

def compute_tapvid_metrics_for_video(
        trajectories_dict: dict,
        occlusions_dict: dict,
        benchmark_data: dict,
        video_idx: int,
        pred_video_sizes=None,
    ):
    """Compute model metrics for TAP-Vid dataset. for a single video.
    Args:
        model_trajectories_dir (str): directory containing model trajectories.
        model_occ_pred_dir (str): directory containing model occlusion predictions.
        benchmark_data (dict): benchmark data dictionary.
        video_idx (int): video index.
        pred_video_sizes (Tuple[int, int]): predicted video sizes. Defaults to None.
    Returns:
        dict: computed metrics.
    """

    for video_config in benchmark_data["videos"]:
        if video_config["video_idx"] == video_idx:
            benchmark_video_data = video_config
            break
    pred_rescale_h = benchmark_video_data['h'] if pred_video_sizes is None else pred_video_sizes[1]
    pred_rescale_w = benchmark_video_data['w'] if pred_video_sizes is None else pred_video_sizes[0]

    video_query_points_list = []
    gt_occluded_list = []
    gt_tracks_list = []
    pred_occluded_list = []
    pred_tracks_list = []
    
    for frame_idx in benchmark_video_data['query_points']:
        
        trajectories = trajectories_dict[frame_idx]
        pred_occluded = occlusions_dict[frame_idx]

        query_points = np.array(benchmark_video_data['query_points'][frame_idx])
        t = np.array([frame_idx] * query_points.shape[0])
        query_points = np.concatenate([t[:, None], query_points], axis=1)

        video_query_points_list.append(query_points)
        gt_tracks_list.append(benchmark_video_data['target_points'][frame_idx])
        gt_occluded_list.append(benchmark_video_data['occluded'][frame_idx])
        pred_tracks_list.append(trajectories)
        pred_occluded_list.append(pred_occluded)
    
    video_query_points = np.concatenate(video_query_points_list, axis=0, dtype=np.float32) # N x 3
    gt_tracks = np.concatenate(gt_tracks_list, axis=0, dtype=np.float32) # N x T x 2
    gt_occluded = np.concatenate(gt_occluded_list, axis=0, dtype=object) # N x T
    pred_tracks = np.concatenate(pred_tracks_list, axis=0, dtype=np.float32) # N x T x 2
    pred_occluded = np.concatenate(pred_occluded_list, axis=0, dtype=object) # N x T

    # rescale and replace (t, x, y) with (t, y, x)
    video_query_points[..., 1] = video_query_points[..., 2] * 256 / (benchmark_video_data['h'])
    video_query_points[..., 2] = video_query_points[..., 1] * 256 / (benchmark_video_data['w'])
    
    gt_tracks[..., 0] *= 256 / (benchmark_video_data['w'])
    gt_tracks[..., 1] *= 256 / (benchmark_video_data['h'])
    
    pred_tracks[..., 0] *= 256 / pred_rescale_w
    pred_tracks[..., 1] *= 256 / pred_rescale_h

    # add batch dimension to each
    video_query_points = video_query_points[None, ...] # 1 x N x 3
    gt_tracks = gt_tracks[None, ...] # 1 x N x T x 2
    gt_occluded = gt_occluded[None, ...] # 1 x N x T
    pred_tracks = pred_tracks[None, ...] # 1 x N x T x 2
    pred_occluded = pred_occluded[None, ...] # 1 x N x T
    
    metrics = compute_tapvid_metrics(video_query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode="strided")
    metrics_clean = {key: value.item() for key, value in metrics.items()} # extract values from numpy arrays
    return metrics_clean

def tracking_single(video_id, module):
    patch_size = 16
    stride = patch_size // 2

    model = copy.deepcopy(module)

    h, w = 476, 854
    if h % patch_size != 0 or w % patch_size != 0:
        print(
            f'Warning: image size ({h}, {w}) is not divisible by patch size {patch_size}')
        h = h // patch_size * patch_size
        w = w // patch_size * patch_size
        print(f'New image size: {h}, {w}')

    # video_root = Path(f'data/tapvid-davis/{video_id}')
    video_root = Path(f'data/davis_480/{video_id}')

    images = []
    for img_fn in sorted((video_root / 'video').glob('*.jpg')):
        images.append(
            np.array(Image.open(img_fn).resize((w, h), Image.LANCZOS)))
    images = np.stack(images)
    images = torch.from_numpy(images).permute(
        0, 3, 1, 2).float().cuda() / 255.0

    features = []
    for image in tqdm(images):
        ph = 1 + (h - patch_size) // stride
        pw = 1 + (w - patch_size) // stride

        # fix the stride
        stride_pair = nn_utils._pair(stride)
        model.model.patch_embed.proj.stride = stride_pair
        # fix the positional encoding code
        model.model.interpolate_pos_encoding = types.MethodType(
            _fix_pos_enc(patch_size, stride_pair), model.model)

        feature = model.model.forward_features(
            imagenet_norm(image[None].cuda()))
        feature = feature[:, 1:]
        feature = feature.reshape(-1, ph, pw,
                                  feature.shape[-1]).permute(0, 3, 1, 2)

        # Only apply refine_conv if it exists in the module
        if hasattr(model, 'refine_conv'):
            feature = model.refine_conv(feature)
            
        features.append(feature)
    features = torch.cat(features)
    dino_tracker = Tracker(
        features, images, dino_patch_size=patch_size, stride=stride)

    anchor_cosine_similarity_threshold = 0.7
    cosine_similarity_threshold = 0.6
    model_inference = ModelInference(
        model=dino_tracker,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=anchor_cosine_similarity_threshold,
        cosine_similarity_threshold=cosine_similarity_threshold,
    )

    rescale_sizes = [dino_tracker.video.shape[-1],
                     dino_tracker.video.shape[-2]]
    benchmark_config = pickle.load(
        open('data/tapvid_davis_data_strided.pkl', "rb"))
    
    for video_config in benchmark_config["videos"]:
        if video_config["video_idx"] == video_id:
            break
    rescale_factor_x = rescale_sizes[0] / video_config['w']
    rescale_factor_y = rescale_sizes[1] / video_config['h']
    query_points_dict = {}

    for frame_idx, q_pts_at_frame in video_config['query_points'].items():
        target_points = video_config['target_points'][frame_idx]
        query_points_at_frame = []
        for q_point in q_pts_at_frame:
            query_points_at_frame.append(
                [rescale_factor_x * q_point[0], rescale_factor_y * q_point[1], frame_idx])
        query_points_dict[frame_idx] = query_points_at_frame

    trajectories_dict = {}
    occlusions_dict = {}
    for frame_idx in tqdm(sorted(query_points_dict.keys()), desc="Predicting trajectories"):
        qpts_st_frame = torch.tensor(
            query_points_dict[frame_idx], dtype=torch.float32, device='cuda')  # N x 3, (x, y, t)
        trajectories_at_st_frame, occlusion_at_st_frame = model_inference.infer(
            query_points=qpts_st_frame, batch_size=None)  # N x T x 3, N x T

        trajectories = trajectories_at_st_frame[..., :2].cpu().detach().numpy()
        occlusions = occlusion_at_st_frame.cpu().detach().numpy()
        trajectories_dict[frame_idx] = trajectories
        occlusions_dict[frame_idx] = occlusions

    # only test video id 0 for now
    metrics = compute_tapvid_metrics_for_video(trajectories_dict=trajectories_dict,
                                               occlusions_dict=occlusions_dict,
                                               video_idx=video_id,
                                               benchmark_data=benchmark_config,
                                               pred_video_sizes=[w, h])
    metrics["video_idx"] = int(video_id)
    return metrics

def tracking(model, num_videos=1):
    metrics_list = []
    for id in range(num_videos):
        metrics = tracking_single(id, module=model)
        metrics_list.append(metrics)
        print(metrics)

    # print(f'summary:')
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index(['video_idx'], inplace=True)
    return metrics_df

def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(
                    target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(
                    target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(
                    target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[
                         (top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(
                    target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[
                         (0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def load_pascal_data(path, size=256, category='cat', split='test', same_view=False):

    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1, 20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    np.random.seed(42)
    files = []
    kps = []
    test_data = pd.read_csv('{}/{}_pairs_pf_{}_views.csv'.format(path,
                            split, 'same' if same_view else 'different'))
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:, 2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    print(f'Number of SPairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id, :]
    src_img_names = np.array(subset_pairs.iloc[:, 0])
    trg_img_names = np.array(subset_pairs.iloc[:, 1])
    # print(src_img_names.shape, trg_img_names.shape)
    point_A_coords = subset_pairs.iloc[:, 3:5]
    point_B_coords = subset_pairs.iloc[:, 5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        point_coords_src = get_points(point_A_coords, i).transpose(1, 0)
        point_coords_trg = get_points(point_B_coords, i).transpose(1, 0)
        src_fn = f'{path}/../{src_img_names[i]}'
        trg_fn = f'{path}/../{trg_img_names[i]}'
        src_size = Image.open(src_fn).size
        trg_size = Image.open(trg_fn).size
        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(
            point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(
            point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None

def semantic_transfer(model, num_cats=None, same_view=False):
    img_size = 640
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    patch_size = 16
    stride = 16
    ph = 1 + (img_size - patch_size) // stride
    pw = 1 + (img_size - patch_size) // stride

    layer_name = 'x_norm_patchtokens'  # choose from x_prenorm, x_norm_patchtokens

    pcks = []
    pcks_05 = []
    pcks_01 = []

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # for pascal

    if num_cats is not None:
        categories = categories[:num_cats]

    for cat in categories:
        files, kps, _ = load_pascal_data(
            'data/PF-dataset-PASCAL', size=img_size, category=cat, same_view=same_view)  # Use same_view parameter

        gt_correspondences = []
        pred_correspondences = []
        for pair_idx in tqdm(range(len(files) // 2)):
            # Load image 1
            img1 = Image.open(files[2*pair_idx]).convert('RGB')
            img1 = resize(img1, img_size, resize=True, to_pil=True, edge=False)
            img1_kps = kps[2*pair_idx]

            # # Get patch index for the keypoints
            img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()

            # Load image 2
            img2 = Image.open(files[2*pair_idx+1]).convert('RGB')
            img2 = resize(img2, img_size, resize=True, to_pil=True, edge=False)
            img2_kps = kps[2*pair_idx+1]

            # Get patch index for the keypoints
            img2_y, img2_x = img2_kps[:, 1].numpy(), img2_kps[:, 0].numpy()

            img1 = torch.from_numpy(
                np.array(img1) / 255.).cuda().float().permute(2, 0, 1)
            img2 = torch.from_numpy(
                np.array(img2) / 255.).cuda().float().permute(2, 0, 1)

            # img1_desc = model.dinov2.forward_features(imagenet_norm(img1[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            img1_desc = model.model.forward_features(
                imagenet_norm(img1[None]))[:, 1:]
            img1_desc = img1_desc.reshape(-1, ph, pw,
                                          img1_desc.shape[-1]).permute(0, 3, 1, 2)

            # img2_desc = model.dinov2.forward_features(imagenet_norm(img2[None]))[layer_name][:, (1 if 'prenorm' in layer_name else 0):]
            img2_desc = model.model.forward_features(
                imagenet_norm(img2[None]))[:, 1:]
            img2_desc = img2_desc.reshape(-1, ph, pw,
                                          img2_desc.shape[-1]).permute(0, 3, 1, 2)

            # Only apply refine_conv if it exists in the model
            if hasattr(model, 'refine_conv'):
                img1_desc = model.refine_conv(img1_desc)
                img2_desc = model.refine_conv(img2_desc)

            ds_size = ((img_size - patch_size) // stride) * stride + 1
            img2_desc = F.interpolate(img2_desc, size=(
                ds_size, ds_size), mode='bilinear', align_corners=True)
            img2_desc = VF.pad(img2_desc, (patch_size // 2, patch_size // 2,
                                           img_size -
                                           img2_desc.shape[2] -
                                           (patch_size // 2),
                                           img_size - img2_desc.shape[3] - (patch_size // 2)), padding_mode='edge')

            vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
            img1_kp_desc = interpolate_features(img1_desc, img1_kps[None, :, :2].cuda(), h=img_size, w=img_size, normalize=True)  # N x F x K
            sim = torch.einsum('nfk,nif->nki', img1_kp_desc, img2_desc.permute(0, 2, 3, 1).reshape(1, img_size * img_size, -1))[0]
            nn_idx = torch.argmax(sim, dim=1)
            nn_x = nn_idx % img_size
            nn_y = nn_idx // img_size
            kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)
            
            gt_correspondences.append(img2_kps[vis][:, [1, 0]])
            pred_correspondences.append(kps_1_to_2[vis][:, [1, 0]])

        gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
        pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
        alpha = torch.tensor([0.1, 0.05, 0.15])
        correct = torch.zeros(len(alpha))

        err = (pred_correspondences - gt_correspondences).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        threshold = alpha * img_size
        correct = err < threshold.unsqueeze(-1)
        correct = correct.sum(dim=-1) / len(gt_correspondences)

        alpha2pck = zip(alpha.tolist(), correct.tolist())
        print(' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                          for alpha, pck_alpha in alpha2pck]))

        pck = correct

        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])

    result = {}
    result['PCK0.05'] = [tensor.item() for tensor in pcks_05]
    result['PCK0.10'] = [tensor.item() for tensor in pcks]
    result['PCK0.15'] = [tensor.item() for tensor in pcks_01]

    metrics_df = pd.DataFrame(result)
    metrics_df['categories'] = categories[:num_cats]
    metrics_df.set_index(['categories'], inplace=True)

    weights = [15, 30, 10, 6, 8, 32, 19, 27, 13, 3,
               8, 24, 9, 27, 12, 7, 1, 13, 20, 15][:num_cats]

    metrics_df['Weighted PCK0.05'] = np.average(metrics_df['PCK0.05'], weights=weights)
    metrics_df['Weighted PCK0.10'] = np.average(metrics_df['PCK0.10'], weights=weights)
    metrics_df['Weighted PCK0.15'] = np.average(metrics_df['PCK0.15'], weights=weights)
    return metrics_df

class TimmEvaluationCallback(pl.Callback):
    # def __init__(self, eval_every_n_epochs: int = 1, eval_methods: list = ['semantic_transfer'], same_view: bool = True):
    def __init__(self, cfg, eval_every_n_epochs: int = 1, eval_methods: list = ['semantic_transfer']):
        super().__init__()
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.eval_every_n_epochs = eval_every_n_epochs
        self.eval_methods = eval_methods
        self.cfg = cfg
        # self.output_dir = Path('evaluation_output/intermediate_results')
        self.output_dir = Path('evaluation_output') / cfg.model / cfg.matcher / cfg.backbone / cfg.dataset / start_time
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        print(f"------------------------------------------------------")
        print(f"(EvaluationCallback) Epoch {trainer.current_epoch + 1}")
        print(f"------------------------------------------------------")

        if hasattr(pl_module, 'batch_metrics'):
            metrics = {
                k: sum(v) / (len(v) + 1e-8) for k, v in pl_module.batch_metrics.items()
            }

            print(f"\n{'='*50}")
            print(f"Epoch {trainer.current_epoch + 1} end")
            print(f"{'='*50}")
            print(f"Average Total Loss: {metrics['total_loss']:.4f}")
            print(f"Average Depth Loss: {metrics['depth_loss']:.4f}")
            print(f"Average Intra Depth Loss: {metrics['intra_depth_loss']:.4f}")
            print(f"Average KL Loss: {metrics['kl_loss']:.4f}")
            print(f"Average AP Loss: {metrics['ap_loss']:.4f}")
            
            trainer.logger.experiment.add_scalar('epoch/total_loss', metrics['total_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/depth_loss', metrics['depth_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/intra_depth_loss', metrics['intra_depth_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/kl_loss', metrics['kl_loss'], trainer.current_epoch)
            trainer.logger.experiment.add_scalar('epoch/ap_loss', metrics['ap_loss'], trainer.current_epoch)
            
            pl_module.batch_metrics = {
                'depth_loss': [],
                'intra_depth_loss': [],
                'kl_loss': [],
                'ap_loss': [],
                'total_loss': []
            }
            print(f"{'='*50}\n")

        if (trainer.current_epoch + 1) % self.eval_every_n_epochs == 0:
            print(f"\nEpoch {trainer.current_epoch + 1} evaluation start")

            pl_module.eval()
            with torch.no_grad():
                epoch_dir = self.output_dir / f"epoch_{trainer.current_epoch + 1}"
                epoch_dir.mkdir(exist_ok=True)

                # Create args object for visualization
                args = types.SimpleNamespace()
                args.exp_name = self.cfg.model
                args.matcher = self.cfg.matcher
                args.backbone = self.cfg.backbone
                args.dataset = self.cfg.dataset

                # Semantic transfer evaluation for both same and different views
                if 'semantic_transfer' in self.eval_methods:
                    print("\nPerforming semantic transfer evaluation...")

                    # --------------------
                    # Same view evaluation
                    # --------------------
                    print("\nSame view evaluation:")
                    metrics_transfer_same = semantic_transfer(pl_module, same_view=True)
                    metrics_transfer_same.to_csv(epoch_dir / 'semantic_transfer_same.csv')

                    mean_metrics_same = metrics_transfer_same.mean()
                    for metric_name, value in mean_metrics_same.items():
                        trainer.logger.experiment.add_scalar(
                            f'semantic_transfer/same_view/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nSame view semantic transfer metrics:")
                    print(mean_metrics_same)

                    # -------------------------
                    # Different view evaluation
                    # -------------------------
                    print("\nDifferent view evaluation:")
                    metrics_transfer_diff = semantic_transfer(pl_module, same_view=False)
                    metrics_transfer_diff.to_csv(epoch_dir / 'semantic_transfer_diff.csv')

                    mean_metrics_diff = metrics_transfer_diff.mean()
                    for metric_name, value in mean_metrics_diff.items():
                        trainer.logger.experiment.add_scalar(
                            f'semantic_transfer/different_view/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nDifferent view semantic transfer metrics:")
                    print(mean_metrics_diff)

                # Object pose estimation evaluation
                if 'pose' in self.eval_methods:
                    print("\nPerforming object pose estimation evaluation...")
                    
                    metrics_pose = oneposepp(pl_module)
                    metrics_pose.to_csv(epoch_dir / 'pose_estimation.csv')
                    
                    mean_metrics_pose = metrics_pose.mean()
                    for metric_name, value in mean_metrics_pose.items():
                        trainer.logger.experiment.add_scalar(
                            f'object_pose/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nObject pose estimation metrics:")
                    print(mean_metrics_pose)

                # Video tracking evaluation
                if 'tracking' in self.eval_methods:
                    print("\nPerforming video tracking evaluation...")
                    
                    metrics_track = tracking(pl_module, num_videos=30)
                    metrics_track.to_csv(epoch_dir / 'tracking.csv')
                    
                    mean_metrics_track = metrics_track.iloc[:, 1:].mean()
                    for metric_name, value in mean_metrics_track.items():
                        trainer.logger.experiment.add_scalar(
                            f'tracking/{metric_name}',
                            value,
                            trainer.current_epoch
                        )
                    print("\nVideo tracking metrics:")
                    print(mean_metrics_track)

                print(f"\nAll results saved at: {epoch_dir}")

            pl_module.train()

def fix_random_seeds(seed=42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def get_masked_patch_cost(cost, mask_patch_1, mask_patch_2=None, eps=1e-8, use_softmax=False, temperature=1.0):
    B, hw, hw2 = cost.shape

    if mask_patch_2 is not None:
        mask_2d = mask_patch_1.unsqueeze(1) * mask_patch_2.unsqueeze(0)
    else:
        # do NOT mask on view 2
        mask_2d = mask_patch_1.unsqueeze(1) * torch.ones_like(mask_patch_1).unsqueeze(0)
    mask_2d = mask_2d.unsqueeze(0).expand(B, hw, hw2)

    masked_cost = cost.clone()
    masked_cost[~mask_2d] = 0.0

    if use_softmax:
        # masked_cost = torch.softmax(masked_cost, dim=-1)
        masked_cost = torch.softmax(masked_cost / temperature, dim=-1, dtype=torch.float32)
    else:
        row_sum = masked_cost.sum(dim=-1, keepdim=True).clamp_min(eps)
        masked_cost = masked_cost / row_sum

    return masked_cost

def kl_divergence_map(mast3r_cost, feat_cost_sim, eps=1e-8):
    mast3r_cost_norm = mast3r_cost.clamp_min(eps)
    feat_cost_sim_norm = feat_cost_sim.clamp_min(eps)

    kl_map = mast3r_cost_norm * torch.log(mast3r_cost_norm / feat_cost_sim_norm)
    # shape: (B, H*W, H*W)

    kl_per_row = kl_map.sum(dim=-1)
    kl_loss = kl_per_row.mean()

    return kl_loss

def get_patch_mask_from_kp_tensor(kp_xy, H, W, patch_size, device=None):
    if device is None:
        device = kp_xy.device

    patch_h = H // patch_size
    patch_w = W // patch_size
    num_patches = patch_h * patch_w

    valid_mask = (kp_xy[:, 0] >= 0) & (kp_xy[:, 0] < W) \
            & (kp_xy[:, 1] >= 0) & (kp_xy[:, 1] < H)
    kp_xy_valid = kp_xy[valid_mask]  # shape: (M, 2), M <= N

    if kp_xy_valid.shape[0] == 0:
        return torch.zeros(num_patches, dtype=torch.bool, device=device)

    x_idx = kp_xy_valid[:, 0].long() // patch_size
    y_idx = kp_xy_valid[:, 1].long() // patch_size

    patch_idx = y_idx * patch_w + x_idx
    # shape: (M,)

    patch_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    patch_mask[patch_idx] = True

    return patch_mask

def pairwise_logistic_ranking_loss(model, pred_scores, gt_depths, depth_threshold=0.0):
    B, N, D = pred_scores.shape
    
    pred_i = pred_scores.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
    pred_j = pred_scores.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)

    depth_i = gt_depths.unsqueeze(2)   # (B, N, 1)
    depth_j = gt_depths.unsqueeze(1)   # (B, 1, N)

    sign_ij = torch.sign(depth_j - depth_i)

    valid_mask = (torch.abs(depth_j - depth_i) > depth_threshold)

    alpha_ij = sign_ij  # (B, N, N)
    # score_diff = (model(pred_j.contiguous().view(B, -1, D)) - model(pred_i.contiguous().view(B, -1, D))).view(B, N, N)  # (B, N, N)
    # score_diff = torch.tanh(score_diff)
    score_diff = model((pred_j - pred_i).view(B, -1, D)).view(B, N, N)  # (B, N, N)
    pairwise_loss = torch.log(1.0 + torch.exp(-alpha_ij * score_diff))

    valid_pairwise_loss = pairwise_loss[valid_mask]
    if valid_pairwise_loss.numel() == 0:
        return torch.tensor(0.0, device=pred_scores.device)
    loss = valid_pairwise_loss.mean()
    return loss

def extract_kp_depth(depth_map, kp, window_size=3):
    B, N, _ = kp.shape
    
    if not torch.is_tensor(depth_map):
        depth_map = torch.tensor(depth_map, device=kp.device, dtype=torch.float)

    depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    H, W = depth_map.shape[-2:]
    half = window_size // 2

    padded = F.pad(depth_map, (half, half, half, half), mode='replicate')  # shape: (1,1,H+2*half, W+2*half)
    
    patches = F.unfold(padded, kernel_size=window_size, stride=1)
    
    # patch_means = patches.mean(dim=1).squeeze(0)  # shape: (H*W)
    patch_means = patches.mean(dim=1) # shape: (B, H*W)
    
    # indices = kp[0, :, 1] * W + kp[0, :, 0]  # shape: (num_points,)
    indices = kp[..., 1] * W + kp[..., 0]  # shape: (B, num_points,)
    
    # avg_depths = patch_means[indices.long()]  # shape: (num_points)
    avg_depths = patch_means.gather(dim=1, index=indices.long())  # shape: (B, num_points)
    
    return avg_depths

def sample_keypoints_nms(mask, conf, N, min_distance, device=None):
    if device is None:
        device = mask.device
    H, W = mask.shape

    score_map = torch.zeros_like(mask, dtype=torch.float32, device=device)
    score_map[mask] = conf[mask]
    
    kernel_size = int(min_distance) * 2 + 1
    pad = kernel_size // 2

    pooled = F.max_pool2d(score_map.unsqueeze(0).unsqueeze(0),
                          kernel_size=kernel_size,
                          stride=1,
                          padding=pad)
    pooled = pooled.squeeze()  # (H, W)

    eps = 1e-6
    nms_mask = (score_map - pooled).abs() < eps
    nms_mask = nms_mask & mask
    
    keypoints = torch.nonzero(nms_mask, as_tuple=False)  # (M, 2)
    
    M = keypoints.shape[0]
    if M == 0:
        return None

    if M > N:
        perm = torch.randperm(M, device=device)[:N]
        sampled_keypoints = keypoints[perm]
    else:
        sampled_keypoints = keypoints
    return sampled_keypoints

def compute_projection(P, points_3d):
    """    
    Args:
        P: (3,4) torch tensor, projection matrix.
        points_3d: (..., 3) tensor of 3D world points.
        
    Returns:
        proj_points: (..., 2) tensor of 2D pixel coordinates.
    """
    orig_shape = points_3d.shape[:-1]
    points_flat = points_3d.view(-1, 3)  # (N,3)
    ones = torch.ones((points_flat.shape[0], 1), dtype=points_flat.dtype, device=points_flat.device)
    points_h = torch.cat([points_flat, ones], dim=1)  # (N,4)
    
    proj_h = P @ points_h.T  # (3,N)
    proj_h = proj_h.T        # (N,3)
    proj_points = proj_h[:, :2] / (proj_h[:, 2:3] + 1e-8)
    return proj_points.view(*orig_shape, 2)

def get_coview_mask(point_map, P_other, image_shape):
    proj_points = compute_projection(P_other, point_map)
    u = proj_points[..., 0]
    v = proj_points[..., 1]
    H_img, W_img = image_shape
    mask = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    return mask

def convert_camera_to_world(point_map, extrinsic):
    R = extrinsic[:, :3]  # (3,3)
    t = extrinsic[:, 3].unsqueeze(0)  # (1,3)
    R_inv = R.t()  # Inverse of R
    world_points = torch.matmul(point_map - t, R_inv)
    return world_points

def get_coview_masks(point_map_view1, point_map_view2, intrinsic1, extrinsic1, intrinsic2, extrinsic2, image_shape):
    world_points_view1 = convert_camera_to_world(point_map_view1, extrinsic1)
    world_points_view2 = convert_camera_to_world(point_map_view2, extrinsic1)
    
    P1 = intrinsic1 @ extrinsic1  # view1: world → view1 image
    P2 = intrinsic2 @ extrinsic2  # view2: world → view2 image
    
    mask1 = get_coview_mask(world_points_view1, P2, image_shape)
    mask2 = get_coview_mask(world_points_view2, P1, image_shape)
    
    return mask1, mask2

# dino patch size is even, so the pixel corner is not really aligned, potential improvements here, borrowed from DINO-Tracker
def interpolate_features(descriptors, pts, h, w, normalize=True, patch_size=14, stride=14):
    last_coord_h = ( (h - patch_size) // stride ) * stride + (patch_size / 2)
    last_coord_w = ( (w - patch_size) // stride ) * stride + (patch_size / 2)
    ah = 2 / (last_coord_h - (patch_size / 2))
    aw = 2 / (last_coord_w - (patch_size / 2))
    bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
    bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
    
    a = torch.tensor([[aw, ah]]).to(pts).float()
    b = torch.tensor([[bw, bh]]).to(pts).float()
    keypoints = a * pts + b
    
    # Expand dimensions for grid sampling
    keypoints = keypoints.unsqueeze(-3)  # Shape becomes [batch_size, 1, num_keypoints, 2]
    
    # Interpolate using bilinear sampling
    interpolated_features = F.grid_sample(descriptors, keypoints, align_corners=True, padding_mode='border')
    
    # interpolated_features will have shape [batch_size, channels, 1, num_keypoints]
    interpolated_features = interpolated_features.squeeze(-2)
    
    return F.normalize(interpolated_features, dim=1) if normalize else interpolated_features

class DepthAwareFeatureFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, use_tanh=True):
        super().__init__()
        self.use_tanh = use_tanh
        
        self.depth_attention = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, features, depths=None):
        if depths is not None:
            # depths: (B, N) -> (B, N, 1)
            depth_embedding = depths.unsqueeze(-1)
            
            # Generate depth-based attention weights
            attention_weights = self.depth_attention(depth_embedding)
            
            # Apply depth-aware attention to features
            depth_modulated_features = features * attention_weights
            
            # Final prediction
            depth_diff = self.fusion_layer(depth_modulated_features)
        else:
            # If no depths provided, just use the features directly
            depth_diff = self.fusion_layer(features)
            
        if self.use_tanh:
            depth_diff = torch.tanh(depth_diff)
            
        return depth_diff.squeeze(-1)
    
class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_dim):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim, bias=False)
        self.relu = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, dim, bias=False)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))
    
class BlockWithAdapter(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x):
        out = self.block(x)
        return out + self.adapter(out)    

class _LoRA_qkv(nn.Module):
    """
    In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module = None,
            linear_b_v: nn.Module = None,
            linear_a_k: nn.Module = None,
            linear_b_k: nn.Module = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        
        qkv[:, :, : self.dim] += new_q

        if self.linear_a_v is not None and self.linear_b_v is not None:
            new_v = self.linear_b_v(self.linear_a_v(x))
            qkv[:, :, -self.dim:] += new_v

        if self.linear_a_k is not None and self.linear_b_k is not None:
            new_k = self.linear_b_k(self.linear_a_k(x))
            qkv[:, :, self.dim:2 * self.dim] += new_k

        return qkv
    


model_configs = {
    'MAE': 'vit_base_patch16_224.mae',
    'CLIP': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', # 'ViT-B-16'
    'DeiT': 'deit3_base_patch16_224.fb_in1k',
    'DINOv2-small': 'dinov2_vits14',
    'DINOv2-base': 'dinov2_vitb14',
    'DINOv2-large': 'dinov2_vitl14',
    'DINOv2-giant': 'dinov2_vitg14',
}
class FinetuneVGGTTIMM(pl.LightningModule):
    def __init__(self,
            device,
            r, 
            backbone_size, 
            datasets,
            ap_loss_weight=1.0,
            depth_loss_weight=1.0,
            intra_depth_loss_weight=1.0,
            kl_loss_weight=1.0,
            ):
        super().__init__()
        self.device = device

        # Save config as hparams
        self.save_hyperparameters()
        self.ap_loss_weight = ap_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.intra_depth_loss_weight = intra_depth_loss_weight
        self.kl_loss_weight = kl_loss_weight

        assert r > 0
        self.embedding_dim = 768 # DINOv2 small : 384, base: 768, large: 1024, giant: 1536,

        self.backbone_name = model_configs[backbone_size]
        print(f"Loading {self.backbone_name}")
        if 'dinov2' in self.backbone_name:
            model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
            self.input_transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            model = timm.create_model(self.backbone_name, pretrained=True, dynamic_img_size=True).cuda().eval()
            data_config = timm.data.resolve_model_data_config(model)
            transforms = timm.data.create_transform(**data_config, is_training=False)
            self.input_transform = transforms.transforms[-1]
        
        self.datasets = datasets
        # --- VGGT Matcher Initialization ---
        print("Loading VGGT matcher...")
        self.matcher = VGGT.from_pretrained("facebook/VGGT-1B").eval() # Set to eval mode
        self.vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.matcher = self.matcher.to(self.device)

        for param in self.matcher.parameters():
            param.requires_grad = False
        # -------------------------------------

        self.w_As = []
        self.w_Bs = []

        for param in model.parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList()
        # --- Adapter initialization for TIMM blocks (indices might need adjustment based on backbone) ---
        adapter_start_idx = 4 # Example: start adapting from the 5th block
        num_adapter_layers = len(model.blocks) - adapter_start_idx
        print(f"Applying LoRA and Adapters to last {num_adapter_layers} blocks starting from index {adapter_start_idx}.")

        for i in range(num_adapter_layers):
            blk_idx = adapter_start_idx + i
            if blk_idx >= len(model.blocks):
                print(f"Warning: Block index {blk_idx} out of range.")
                continue
            blk = model.blocks[blk_idx]

            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

            adapter = Adapter(dim=self.embedding_dim, bottleneck_dim=64)
            blk_with_adapter = BlockWithAdapter(blk, adapter)
            model.blocks[blk_idx] = blk_with_adapter # Use modified block
            self.adapters.append(adapter)
        # -------------------------------------------------------------------------------------------
        self.reset_parameters()

        self.model = model
        # self.downsample_factor = model.patch_embed.patch_size[0] # Use backbone's downsample factor
        self.downsample_factor = 8

        self.refine_conv = nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        # --- Thresholds (can be tuned) ---
        self.thres3d_neg = 0.1
        # --------------------------------

        self.patch_size = model.patch_embed.patch_size[0]
        self.target_res = 640 # Or derive from data_config if needed

        self.min_conf_thr = 10 # Confidence threshold for keypoint filtering (if using VGGT confidence)
        self.count = 0

        self.depth_diff_head = DepthAwareFeatureFusion(input_dim=self.embedding_dim, use_tanh=True)

        # --- VGGT Specific Parameters ---
        self.resize_patch_size = self.matcher.aggregator.patch_size
        self.init_temperature = 1.0
        self.final_temperature = 1.0
        self.matcher.aggregator.temperature = self.init_temperature
        # --------------------------------
        get_model_para(self.matcher)
        get_model_para(self)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        checkpoint['state_dict'] = {
            'refine_conv': self.refine_conv.state_dict(),
        }
        
        depth_diff_head = {
            'depth_diff_head': self.depth_diff_head.state_dict()
        }

        checkpoint.update(a_tensors)
        checkpoint.update(b_tensors)
        checkpoint.update(depth_diff_head)
        
        adapter_tensors = {f"adapter_{i:03d}": adapter.state_dict() for i, adapter in enumerate(self.adapters)}
        checkpoint.update(adapter_tensors)
        
        
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.refine_conv.load_state_dict(checkpoint['state_dict']['refine_conv'])
        
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = checkpoint[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        self.depth_diff_head.load_state_dict(checkpoint['depth_diff_head'])
        
        for i, adapter in enumerate(self.adapters):
            saved_key = f"adapter_{i:03d}"
            adapter.load_state_dict(checkpoint[saved_key])

        self.loaded = True
            
    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(42)
        return torch.utils.data.DataLoader(
            dataset=self.datasets,
            batch_size=1,
            shuffle=False,
            num_workers=10,
            pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id),
            generator=g,
        )

    def get_intermediate_feature(
        self,
        rgbs: torch.Tensor,
        pts=None,
        n=[0,1,2,3],
        reshape: bool = False,
        return_class_token: bool = False,
        normalize: bool = True,
    ):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        resize_factor = [(patch_w * self.patch_size) / rgbs.shape[-1], (patch_h * self.patch_size) / rgbs.shape[-2]]
        
        pts = pts * torch.tensor(resize_factor).to(pts.device)
        
        outputs = self.model._intermediate_layers(self.input_transform(rgb_resized), n)
        if normalize:
            outputs = [self.model.norm(out) for out in outputs]
        if return_class_token:
            prefix_tokens = [out[:, 0] for out in outputs]

        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        if reshape:
            results = []
            for out in outputs:
                res = out.reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()
                res_kp = interpolate_features(res, pts, h=patch_h * self.patch_size, w=patch_w * self.patch_size, 
                                              patch_size=self.patch_size, stride=self.patch_size, normalize=False).permute(0, 2, 1)
                results.append(res_kp)
            
            outputs = torch.stack(results, dim=0).mean(dim=0)

            if return_class_token:
                results_prefix = []
                for prefix_token in prefix_tokens:
                    results_prefix.append(prefix_token.unsqueeze(0).repeat(pts.size(1), 1))

                prefix_tokens = torch.stack(results_prefix, dim=0).mean(dim=0)

        if return_class_token:
            return outputs, prefix_tokens
        return outputs
    
    def get_feature(self, rgbs, pts, normalize=True, global_feature=False):
        tgt_size = (int(rgbs.shape[-2] * self.target_res / rgbs.shape[-1]), self.target_res)
        if rgbs.shape[-2] > rgbs.shape[-1]:
            tgt_size = (self.target_res, int(rgbs.shape[-1] * self.target_res / rgbs.shape[-2]))
        
        patch_h, patch_w = tgt_size[0] // self.downsample_factor, tgt_size[1] // self.downsample_factor
        rgb_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))
        
        resize_factor = [(patch_w * self.patch_size) / rgbs.shape[-1], (patch_h * self.patch_size) / rgbs.shape[-2]]
        
        pts = pts * torch.tensor(resize_factor).to(pts.device)
        
        if global_feature:
            result = self.model.forward_features(self.input_transform(rgb_resized))
            global_feat, result = result[:, 0], result[:, 1:]
        else:    
            if 'dinov2' in self.backbone_name:
                result = self.model.forward_features(self.input_transform(rgb_resized)) 
            else:
                result = self.model.forward_features(self.input_transform(rgb_resized))[:, 1:]
        
        if 'dinov2' in self.backbone_name:
            feature = result['x_norm_patchtokens'].reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)
        else:
            feature = result.reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)

        feature = self.refine_conv(feature)
            
        feature = interpolate_features(feature, pts, h=patch_h * self.patch_size, w=patch_w * self.patch_size, patch_size=self.patch_size, stride=self.patch_size, normalize=False).permute(0, 2, 1)
        if normalize:
            feature = F.normalize(feature, p=2, dim=-1)
        
        if global_feature:
            return feature, global_feat

        return feature

    

    def get_feature_cost(self, rgbs, normalize=True, resize=True):
        B, _, H, W = rgbs.shape
        patch_h = H // self.resize_patch_size
        patch_w = W // self.resize_patch_size

        rgbs_resized = functional.resize(rgbs, (patch_h * self.patch_size, patch_w * self.patch_size))

        outputs = self.model._intermediate_layers(self.input_transform(rgbs_resized), [7]) 
        if normalize:
            outputs = [self.model.norm(out) for out in outputs]

        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        results = []
        for out in outputs:
            res = out.reshape(rgbs.shape[0], patch_h, patch_w, -1)
            results.append(res)
        
        feature = torch.stack(results, dim=0).mean(dim=0)

        return feature

    def extract_vggt_features(self, rgb_vggt, batch_idx=None):
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.vggt_dtype):
                images = rgb_vggt  # add batch dimension
                aggregated_tokens_list, ps_idx, attn = self.matcher.aggregator(images)  # attn (B*S, num_heads, P, P) 全局注意力权重矩阵

            # Predict Cameras
            pose_enc = self.matcher.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = self.matcher.depth_head(aggregated_tokens_list, images, ps_idx)

            # Predict Point Maps
            point_map, point_conf = self.matcher.point_head(aggregated_tokens_list, images, ps_idx)
                
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
            
            # point_map_view_1, point_map_view_2 = point_map[0, 0], point_map[0, 1]
            point_map_view_1 = torch.tensor(point_map_by_unprojection[0], dtype=torch.float32).to(self.device)
            point_map_view_2 = torch.tensor(point_map_by_unprojection[1], dtype=torch.float32).to(self.device)
            point_conf_view_1, point_conf_view_2 = point_conf[0, 0], point_conf[0, 1]
            extrinsic_1, extrinsic_2 = extrinsic[0, 0], extrinsic[0, 1]
            intrinsic_1, intrinsic_2 = intrinsic[0, 0], intrinsic[0, 1]
            depth_pred_1, depth_pred_2 = depth_map[0, 0].squeeze(-1), depth_map[0, 1].squeeze(-1)

            image_shape = tuple(rgb_vggt.shape[-2:])
            
            cost_1, cost_2 = attn.chunk(2, dim=0)   # 拆分为两个 (1, num_heads, P, P) 的自注意力权重 （视图1，视图2）
            cost_1 = cost_1.mean(dim=1)             # 多头注意力关注不同特征，取平均得到更鲁棒的相似度矩阵 (B, P, P)
            cost_2 = cost_2.mean(dim=1)
            
        return {
            'point_map_view_1': point_map_view_1,
            'point_map_view_2': point_map_view_2,
            'point_conf_view_1': point_conf_view_1, 
            'point_conf_view_2': point_conf_view_2,
            'extrinsic_1': extrinsic_1,
            'extrinsic_2': extrinsic_2,
            'intrinsic_1': intrinsic_1,
            'intrinsic_2': intrinsic_2,
            'depth_pred_1': depth_pred_1,
            'depth_pred_2': depth_pred_2,
            'image_shape': image_shape,
            'cost_1': cost_1,
            'cost_2': cost_2,
            'aggregated_tokens_list': aggregated_tokens_list,
            'images': images,
            'ps_idx': ps_idx
        }
    
    def sample_keypoints(self, vggt_features, num_keypoints=300, min_distance=5):
        point_map_view_1 = vggt_features['point_map_view_1']
        point_map_view_2 = vggt_features['point_map_view_2']
        point_conf_view_1 = vggt_features['point_conf_view_1']
        intrinsic_1 = vggt_features['intrinsic_1']
        extrinsic_1 = vggt_features['extrinsic_1']
        intrinsic_2 = vggt_features['intrinsic_2']
        extrinsic_2 = vggt_features['extrinsic_2']
        image_shape = vggt_features['image_shape']
        aggregated_tokens_list = vggt_features['aggregated_tokens_list']
        images = vggt_features['images']
        ps_idx = vggt_features['ps_idx']
        
        mask_1, mask_2 = get_coview_masks(point_map_view_1, point_map_view_2,
                                    intrinsic_1, extrinsic_1,
                                    intrinsic_2, extrinsic_2,
                                    image_shape)
        
        # 在mask为True的有效区域内，通过非极大值抑制（NMS）筛选出置信度图conf的局部最大值，最终返回最多300个关键点的坐标
        sampled_kp_1 = sample_keypoints_nms(mask_1, point_conf_view_1, N=num_keypoints, min_distance=min_distance, device=self.device)
        
        if sampled_kp_1 is None:
            print("No keypoints found in the first view.")
            return None, None, None, None, None

        sampled_kp_1 = sampled_kp_1[:, [1, 0]].int()  # (row, col) -> (x, y)
        sampled_kp_2, vis_score, conf_score = self.matcher.track_head(aggregated_tokens_list, images, ps_idx, query_points=sampled_kp_1[None])
        sampled_kp_2 = sampled_kp_2[-1][0][1].int()  # (x, y)
        
        mh, mw = image_shape
        valid_kp_1 = (sampled_kp_1[:, 0] >= 3) & (sampled_kp_1[:, 0] < int(mw) - 3) & (sampled_kp_1[:, 1] >= 3) & (sampled_kp_1[:, 1] < int(mh) - 3)
        valid_kp_2 = (sampled_kp_2[:, 0] >= 3) & (sampled_kp_2[:, 0] < int(mw) - 3) & (sampled_kp_2[:, 1] >= 3) & (sampled_kp_2[:, 1] < int(mh) - 3)
        valid_kp = valid_kp_1 & valid_kp_2
        
        kp_1 = sampled_kp_1[valid_kp].float().unsqueeze(0).to(self.device)
        kp_2 = sampled_kp_2[valid_kp].float().unsqueeze(0).to(self.device)
        
        return kp_1, kp_2, valid_kp, mask_1, mask_2
    
    def update_temperature(self):
        total_epochs = self.trainer.max_epochs if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs > 0 else 100
        current_epoch = self.current_epoch
        ratio = min(current_epoch / total_epochs, 1.0)
        
        new_temp = self.init_temperature * (1 - ratio) + self.final_temperature * ratio
    
        self.matcher.aggregator.temperature = new_temp
    
    
    
    def on_train_batch_end(self, *args, **kwargs):
        self.update_temperature()

    def calculate_depth_loss(self, vggt_features, rgb_1_resized, rgb_2_resized, kp_1, kp_2, indices=[4,5,6,7]):
        depth_pred_1 = vggt_features['depth_pred_1']
        depth_pred_2 = vggt_features['depth_pred_2']
        
        kp_feat_1 = self.get_intermediate_feature(rgb_1_resized, pts=kp_1, n=indices, reshape=True, normalize=True)
        kp_feat_2 = self.get_intermediate_feature(rgb_2_resized, pts=kp_2, n=indices, reshape=True, normalize=True)

        kp_depth_1 = extract_kp_depth(depth_pred_1, kp_1)
        kp_depth_2 = extract_kp_depth(depth_pred_2, kp_2)
        
        kp_depth_diff = kp_depth_1 - kp_depth_2
        kp_feature_diff = kp_feat_1 - kp_feat_2
        pred_depth_diff = self.depth_diff_head(kp_feature_diff)
        
        depth_loss = F.l1_loss(pred_depth_diff, torch.tanh(kp_depth_diff).detach())
        
        pairwise_loss_1 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_1, kp_depth_1, depth_threshold=0.05)
        pairwise_loss_2 = pairwise_logistic_ranking_loss(self.depth_diff_head, kp_feat_2, kp_depth_2, depth_threshold=0.05)
        intra_depth_loss = (pairwise_loss_1 + pairwise_loss_2) / 2
        
        return depth_loss, intra_depth_loss


    def calculate_cost_loss(self, rgb_1_resized, rgb_2_resized, vggt_cost_1, vggt_cost_2, kp_1=None, kp_2=None, mask_1=None, mask_2=None):
        B, C, H, W = rgb_1_resized.shape
        
        feat_cost_1 = self.get_feature_cost(rgb_1_resized, normalize=False, resize=True)
        feat_cost_2 = self.get_feature_cost(rgb_2_resized, normalize=False, resize=True)
        
        patch_h = H // self.resize_patch_size
        patch_w = W // self.resize_patch_size
        
        if kp_1 is not None and kp_2 is not None:
            kp_xy_1 = kp_1[0]  # (N, 2) (x,y)
            mask_patch_1 = get_patch_mask_from_kp_tensor(kp_xy_1, H, W, self.patch_size)
            
            kp_xy_2 = kp_2[0]  # (N, 2)
            mask_patch_2 = get_patch_mask_from_kp_tensor(kp_xy_2, H, W, self.patch_size)
        else:   # 使用 最近邻插值 将掩码对齐至特征图尺寸 (patch_h, patch_w)，保持掩码的布尔性质
            mask_patch_1 = F.interpolate(mask_1.unsqueeze(0).unsqueeze(0).float(), size=(patch_h, patch_w), mode='nearest').squeeze(0).bool()
            mask_patch_1 = mask_patch_1.view(-1)    # (patch_h, patch_w) → (patch_h * patch_w,)

            mask_patch_2 = F.interpolate(mask_2.unsqueeze(0).unsqueeze(0).float(), size=(patch_h, patch_w), mode='nearest').squeeze(0).bool()
            mask_patch_2 = mask_patch_2.view(-1)
        
        feat_cost_1 = feat_cost_1.view(1, patch_h * patch_w, -1)  # (B, H*W, C)
        feat_cost_2 = feat_cost_2.view(1, patch_h * patch_w, -1)  # (B, H*W, C)
        
        feat_cost_1 = F.normalize(feat_cost_1, p=2, dim=-1)       # L2 归一化 确保特征向量模长为1，使得点积表示 余弦相似度
        feat_cost_2 = F.normalize(feat_cost_2, p=2, dim=-1)
        
        feat_cost_1_to_2 = torch.bmm(feat_cost_1, feat_cost_2.transpose(-1, -2))  # (B, H*W, H*W)
        feat_cost_2_to_1 = torch.bmm(feat_cost_2, feat_cost_1.transpose(-1, -2))  # (B, H*W, H*W)
        
        feat_cost_1_to_2 = torch.nn.functional.softmax(feat_cost_1_to_2, dim=-1)  # Softmax 归一化：将相似度转换为概率分布（每行和为1）
        feat_cost_2_to_1 = torch.nn.functional.softmax(feat_cost_2_to_1, dim=-1)
        
        masked_cost_1 = get_masked_patch_cost(vggt_cost_1, mask_patch_1, mask_patch_2=None)
        masked_cost_2 = get_masked_patch_cost(vggt_cost_2, mask_patch_2, mask_patch_2=None)
        
        masked_feat_cost_1_to_2 = get_masked_patch_cost(feat_cost_1_to_2, mask_patch_1, mask_patch_2=None)
        masked_feat_cost_2_to_1 = get_masked_patch_cost(feat_cost_2_to_1, mask_patch_2, mask_patch_2=None)
        
        kl_loss_1 = kl_divergence_map(masked_cost_1, masked_feat_cost_1_to_2)
        kl_loss_2 = kl_divergence_map(masked_cost_2, masked_feat_cost_2_to_1)
        
        kl_loss = (kl_loss_1 + kl_loss_2) / 2
        
        return kl_loss


    def calculate_matching_loss(self, rgb_1_resized, rgb_2_resized, kp_1, kp_2, point_map_view_1, point_map_view_2):
        desc_1 = self.get_feature(rgb_1_resized, kp_1, normalize=True)  # (B, N, C)
        desc_2 = self.get_feature(rgb_2_resized, kp_2, normalize=True)  # (B, N, C)
        
        pts3d_1 = point_map_view_1[kp_1[...,1].long(), kp_1[...,0].long()]  # (B, N, 3)
        pts3d_2 = point_map_view_2[kp_2[...,1].long(), kp_2[...,0].long()]  # (B, N, 3)
        
        pos_idxs = torch.stack([
            torch.zeros(desc_1.size(1), dtype=torch.long, device=self.device),  # [0, 0, ..., 0]
            torch.arange(desc_1.size(1), device=self.device),                   # [0, 1, ..., N-1]
            torch.arange(desc_2.size(1), device=self.device)                    # [0, 1, ..., N-1]
        ], dim=1)  # (N, 3) each row [0, i, i]
        
        eye_mask = torch.eye(desc_1.size(1), device=self.device).bool().unsqueeze(0)    # (1, N, N) only diag elements are 'True'
        neg_mask = (torch.cdist(pts3d_1, pts3d_2) > self.thres3d_neg) & ~eye_mask       # (B, N, N)
        
        sim = torch.bmm(desc_1, desc_2.transpose(-1, -2))  # (B, N, N) similarities between features of kp_1[i] and kp_2[j]
        
        pos_sim = sim[pos_idxs[:,0], pos_idxs[:,1], pos_idxs[:,2]]      # (N) similarities of diag elements (positive pairs)
        rpos = sigmoid(1. - pos_sim, temp=0.01) + 1  # (N)
        rall = rpos + torch.sum(
            sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - 1., temp=0.01)  # the first N rows of sim (the whole sim)
            * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),           # the first N rows of neg_mask (the whole neg_mask)
            dim=-1
        )
        ap1 = rpos / rall
        
        rpos = sigmoid(1. - pos_sim, temp=0.01) + 1
        rall = rpos + torch.sum(
            sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - pos_sim[:, None], temp=0.01) # pos_sim[:, None] (N, 1)
            * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),
            dim=-1
        )
        ap2 = rpos / rall
        
        ap = (ap1 + ap2) / 2
        ap_loss = torch.mean(1. - ap)
        
        return ap_loss


    def training_step(self, batch, batch_idx):
        rgb_1, rgb_vggt = batch['rgb_1'], batch['rgb_vggt']
        rgb_2 = batch['rgb_2']

        vggt_features = self.extract_vggt_features(rgb_vggt, batch_idx=batch_idx)
        
        kp_1, kp_2, valid_kp, mask_1, mask_2 = self.sample_keypoints(vggt_features, num_keypoints=300, min_distance=5)
        
        if kp_1 is None or kp_2 is None:
            loss = torch.tensor(0., device=self.device, requires_grad=True)
            self.log('loss', loss, prog_bar=True)
            return loss

        mh, mw = vggt_features['image_shape']
        rgb_1_resized = functional.resize(rgb_1, (mh, mw))
        rgb_2_resized = functional.resize(rgb_2, (mh, mw))

        if kp_1.size(1) == 0 or kp_2.size(1) == 0:
            loss = torch.tensor(0., device=self.device, requires_grad=True)
            self.log('loss', loss, prog_bar=True)
            return loss

        depth_loss, intra_depth_loss = self.calculate_depth_loss(
            vggt_features, rgb_1_resized, rgb_2_resized, kp_1, kp_2
        )

        kl_loss = self.calculate_cost_loss(
            rgb_1_resized, rgb_2_resized, vggt_features['cost_1'], vggt_features['cost_2'], 
            mask_1=mask_1, mask_2=mask_2
        )

        ap_loss = self.calculate_matching_loss(
            rgb_1_resized, rgb_2_resized, kp_1, kp_2, 
            vggt_features['point_map_view_1'], vggt_features['point_map_view_2']
        )

        loss = (self.ap_loss_weight * ap_loss + 
                self.depth_loss_weight * depth_loss + 
                self.intra_depth_loss_weight * intra_depth_loss + 
                self.kl_loss_weight * kl_loss)

        self.log('loss', loss, prog_bar=True)
        self.log('depth_loss', depth_loss, prog_bar=True)
        self.log('intra_depth_loss', intra_depth_loss, prog_bar=True)
        self.log('kl_loss', kl_loss, prog_bar=True)
        self.log('ap_loss', ap_loss, prog_bar=True)
        
        if not hasattr(self, 'batch_metrics'):
            self.batch_metrics = {
                'depth_loss': [],
                'intra_depth_loss': [],
                'kl_loss': [],
                'ap_loss': [],
                'total_loss': []
            }
        
        self.batch_metrics['depth_loss'].append(depth_loss.item())
        self.batch_metrics['intra_depth_loss'].append(intra_depth_loss.item())
        self.batch_metrics['kl_loss'].append(kl_loss.item())
        self.batch_metrics['ap_loss'].append(ap_loss.item())
        self.batch_metrics['total_loss'].append(loss.item())
        
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW([layer.weight for layer in self.w_As]
                                 + [layer.weight for layer in self.w_Bs]
                                 + list(self.refine_conv.parameters())
                                 + list(self.depth_diff_head.parameters())
                                 +  list(self.adapters.parameters())
                                 , lr=1e-5, weight_decay=1e-4)



class ObjaverseVGGTDataset(Dataset):
    def __init__(self, root, num) -> None:
    # def __init__(self, root) -> None:
        super().__init__()
        self.root = Path(root)

        scale_x = 512 / 512
        scale_y = 384 / 512

        self.intrinsic = np.array([
                    [16 * 512 * scale_x / 32., 0, 256 * scale_x],
                    [0, 16 * 512 * scale_y / 32., 256 * scale_y],
                    [0, 0, 1]
                ])

        with open('data/10k.txt', 'r') as file:
            txt_obj_names = [line.strip() for line in file.readlines()]

        self.obj_names = txt_obj_names[:num]
        self.num_objects = len(self.obj_names)
        self.obj_rgb_max_idx = {obj_name: self.get_rgb_max_idx(obj_name) for obj_name in self.obj_names}

    def get_rgb_max_idx(self, obj_name):
        regex_path = os.path.join(self.root, obj_name, 'color_*.png')
        max_idx = 0
        for path in glob.glob(regex_path):
            idx = int(path.split('_')[-1].split('.')[0])
            max_idx = max(max_idx, idx)
        return max_idx
    
    def get_item(self, index, suffix='', obj_name=None, i=None):
        if index >= len(self):
            raise IndexError('index out of range')
        if obj_name is None:
            while True:
                obj_name = np.random.choice(self.obj_names)
                if self.obj_rgb_max_idx[obj_name] > 1:
                    break
        if i is None:
            i = np.random.choice(self.obj_rgb_max_idx[obj_name])
        rgb_path = self.root / obj_name / f'color_{i:06d}.png'
        rgb = cv2.imread(str(rgb_path))[..., ::-1].copy()

        depth_path = self.root / obj_name / f'depth_{i:06d}.png'
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).copy()
        depth[depth == 0] = 5000
        depth[depth > 5000] = 5000   

        return {
            f'obj_name_{suffix}': obj_name,
            f'rgb_{suffix}': np.moveaxis((rgb / 255.).astype(np.float32), -1, 0),
            f'rgb_path_{suffix}': str(rgb_path),
            f'pose_idx_{suffix}': i,
            f'depth_{suffix}': (depth / 5000.).astype(np.float32),
            f'depth_path_{suffix}': str(depth_path),
        }
        
    def __getitem__(self, idx):
        try:
            res1 = self.get_item(idx, '1')
            obj_name_1 = res1['obj_name_1']
            pose_idx = res1[f'pose_idx_1']
            i = np.random.choice(self.obj_rgb_max_idx[obj_name_1])
            while i == pose_idx:
                i = np.random.choice(self.obj_rgb_max_idx[obj_name_1])
            res2 = self.get_item(idx, '2', obj_name_1, i)

            img1_path = res1['rgb_path_1']
            img2_path = res2['rgb_path_2']

            rgb_vggt = load_and_preprocess_images([str(img1_path), str(img2_path)])

            res = {**res1, **res2}
            
            res.update({'rgb_vggt': rgb_vggt})
            res.update({'intrinsic': self.intrinsic})

        except Exception as e:
            # print(e)
            res = self[(idx + 1) % len(self)]
        return res
    
    def __len__(self):
        # return len(self.obj_names)
        return 100



class AugmentedCustomObjaverseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        Args:
            dataset (Dataset): Instance of GoogleObjectsDataset.
            coco_root (str): Directory with all the images from COCO.
            coco_ann_file (str): Path to the JSON file with COCO annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = dataset
        self.color_augs = A.Compose([
            A.GaussianBlur(blur_limit=(1, 3)),
            A.ISONoise(),
            A.GaussNoise(),
            A.CLAHE(),  # could probably be moved to the post-crop augmentations
            A.RandomBrightnessContrast(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        for img_idx in [1, 2]:
            obj_image = (np.moveaxis(data[f'rgb_{img_idx}'], 0, -1) * 255).astype(np.uint8)
            obj_image = self.color_augs(image=obj_image)['image']

            # Update the dataset entry
            data[f'rgb_{img_idx}'] = np.moveaxis((obj_image / 255.).astype(np.float32), -1, 0)

        return data
    


def get_dataset(dataset_name):
    assert dataset_name == 'objaverse'
    return AugmentedCustomObjaverseDataset(ConcatDataset([ObjaverseVGGTDataset('data/objaverse_renderings', 10_000)]))

@hydra.main(
    config_path='../config',
    config_name='finetune_timm_vggt_scannetpp',
    # config_name='finetune_timm_vggt_objaverse',
    version_base='1.2',
)
def main(cfg):
    rank = 2
    # Add evaluation methods as a list in the config
    eval_methods = cfg.get(
        'evaluation_methods', ['semantic_transfer']
    )

    # is_dev_mode = True
    is_dev_mode = False
    limit_batches = 2 if is_dev_mode else None
    # limit_batches = 1 if is_dev_mode else None  # Naive CLIP
    # ---------------------------------------

    fix_random_seeds()
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"Output directory: {output_dir}")

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = TensorBoardLogger(
        output_dir,
        name='experiment_logs',
        version=f"{start_time}_{cfg.get('experiment_version', 'v1')}",
        default_hp_metric=False
    )
    logger.log_hyperparams({
        'notes': 'Your experiment notes here',
        'start_time': start_time,
        'model_type': 'ViT-B-16',
        'dataset': cfg.dataset,
        'batch_size': 'default'
    })


    dataset_instance = get_dataset(cfg.dataset, 'vggt')
    pl_module = FinetuneVGGTTIMM(device=f"cuda:{rank}", r=4, backbone_size='ViT-B-16', datasets=dataset_instance)

    evaluation_callback = TimmEvaluationCallback(
        cfg,
        eval_every_n_epochs=10,
        eval_methods=eval_methods
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints', start_time),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true',
        max_epochs=500,
        limit_train_batches=limit_batches,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=[
            evaluation_callback,
            checkpoint_callback,
        ],
    )
    print("Starting training...")
    trainer.fit(pl_module)
    print("Training finished.")
    pass


if __name__ == '__main__':
    main()