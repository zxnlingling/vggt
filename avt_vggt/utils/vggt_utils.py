import os
import cv2
import torch
import argparse
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from torchvision import transforms
from vggt.utils.load_fn import load_and_preprocess_images
from .env_utils import (CAMERAS, IMAGE_SIZE, EPISODE_FOLDER, CAMERA_FRONT, 
                        IMAGE_RGB, IMAGE_FORMAT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST)
from typing import List, Tuple, Union, Optional, Dict
from .mvt_utils import select_feat_from_hm
from vggt.heads.dpt_head import _make_scratch, _make_fusion_block, custom_interpolate
from vggt.training.loss import reg_loss, normalize_pointcloud
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

DATA_FOLDER = "/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum"
PRE_CKPT = "/fs-computility/efm/lvqi/projects/colosseum/SAM2Act/sam2Act_COLOSSEUM/third_libraries/vggt/model.pt"


def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.
    
    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)
        
    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        if not isinstance(color_jitter, dict):
            raise ValueError("color_jitter must be a dictionary or None")
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None


def get_model_para(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    param_size = 0
    trainable_param_size = 0
    param_num = 0
    trainable_para_num = 0
    for param in model.parameters():
        param_num += param.nelement() 
        param_size += param.nelement() * param.element_size()
        trainable_para_num += param.nelement() if param.requires_grad else 0
        trainable_param_size += param.nelement() * param.element_size() if param.requires_grad else 0
        
    
    print(f'{model.__class__.__name__}\'s parameter size: {param_size/1024/1024}MB')
    print(f'{model.__class__.__name__}\'s trainable parameter size: {trainable_param_size/1024/1024}MB')
    
    print(f'{model.__class__.__name__}\'s parameter num: {param_num/1000/1000}M')
    print(f'{model.__class__.__name__}\'s trainable parameter num: {trainable_para_num/1000/1000}M')


def get_pc_img_feat_with_positions(obs, pcd, original_rgbs):
    """
    preprocess both the point cloud and rgb data to shape (b, H * W * 4, 3)
    additionally return pixel positions (b, H * W * 4, 3) (view_idx, y, x)

    """
    bs = obs[0][0].shape[0]
    
    # concatenating the points from all the cameras
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    img_feat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1)
    img_feat = (img_feat + 1) / 2
    
    num_views = len(pcd)
    img_height = pcd[0].shape[2]
    img_width = pcd[0].shape[-1]
    total_pixels = img_height * img_width * num_views
    
    # create position for each pixel 
    positions = []
    for view_idx in range(num_views):
        # create pixel mesh for the current view 
        y_coords, x_coords = torch.meshgrid(
            torch.arange(img_height),
            torch.arange(img_width),
            indexing='ij'
        )
        # add view indices
        view_indices = torch.full((img_height, img_width), view_idx)
        view_positions = torch.stack([view_indices, y_coords, x_coords], dim=-1)    
        positions.append(view_positions)
    
    # concat pixel positions of all views
    pixel_positions = torch.cat([p.reshape(-1, 3) for p in positions], 0)
    
    # copy pixel positions for the whole batch
    pixel_positions = pixel_positions.unsqueeze(0).repeat(bs, 1, 1).to(pc.device)

    colors = []
    camera_names = list(original_rgbs.keys())
    for b in range(bs):
        batch_colors = []
        for view_idx in range(num_views):
            camera_name = camera_names[view_idx]
            rgb_img = original_rgbs[camera_name][b]     # (3, H, W)
            view_mask = (pixel_positions[b, :, 0] == view_idx)
            view_pos = pixel_positions[b, view_mask]    # (N_view, 3)

            ys = view_pos[:, 1].long()
            xs = view_pos[:, 2].long()

            valid_mask = (xs >= 0) & (xs < img_width) & (ys >= 0) & (ys < img_height)

            view_colors = torch.zeros(len(ys), 3, device=rgb_img.device)
            
            if valid_mask.any():
                valid_colors = rgb_img[:, ys[valid_mask], xs[valid_mask]].t()
                view_colors[valid_mask] = valid_colors
            
            batch_colors.append(view_colors)
        
        batch_colors = torch.cat(batch_colors, dim=0)
        colors.append(batch_colors)
    
    return pc, img_feat, pixel_positions, colors


def move_pc_in_bound_with_positions(pc, img_feat, bounds, colors, pixel_positions=None, no_op=False):
    """
    :param pc and colors: tensor (b, H * W * 4, 3) or list of tensors
    :param img_feat: (b, H * W * 4, channels)
    :param bounds: scene bounds [x_min, y_min, z_min, x_max, y_max, z_max]
    :param pixel_positions: (b, H * W * 4, 3) (view_idx, y, x)
    :param no_op: no operation

    """
    if no_op:
        return pc, img_feat, pixel_positions, colors if pixel_positions is not None else None
    
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    
    assert colors is not None
    # list
    new_pc = []
    new_img_feat = []
    new_colors = []
    new_positions = [] if pixel_positions is not None else None
    
    for i in range(len(pc)):
        inv_pnt = (
            (pc[i][:, 0] < x_min)
            | (pc[i][:, 0] > x_max)
            | (pc[i][:, 1] < y_min)
            | (pc[i][:, 1] > y_max)
            | (pc[i][:, 2] < z_min)
            | (pc[i][:, 2] > z_max)
            | torch.isnan(pc[i][:, 0])
            | torch.isnan(pc[i][:, 1])
            | torch.isnan(pc[i][:, 2])
        )
        
        new_pc.append(pc[i][~inv_pnt])
        new_img_feat.append(img_feat[i][~inv_pnt])
        new_colors.append(colors[i][~inv_pnt])
        
        if pixel_positions is not None:
            new_positions.append(pixel_positions[i][~inv_pnt])
                
    pc = new_pc
    img_feat = new_img_feat
    colors = new_colors
    pixel_positions = new_positions
    
    return pc, img_feat, pixel_positions, colors


def restore_cropped_image(img_feat, pixel_positions, points=None):
    """
    preprocess the cropped img_feat to shape (b, v, 3, H, W)
    params:
        img_feat: list of b tensors, each shape [n_points, 3]
        points: list of b tensors, each shape [n_points, 3]
        pixel_positions: list of b tensors, each shape [n_points, 3] 
                         (view_idx, y, x)
    """
    batch_size = len(img_feat)
    channels = img_feat[0].shape[1]
    num_views = len(CAMERAS)
    img_height = IMAGE_SIZE
    img_width = IMAGE_SIZE
    
    restored = torch.zeros(
        batch_size, num_views, channels, img_height, img_width,
        device=img_feat[0].device, dtype=img_feat[0].dtype
    )
    restored_points = torch.zeros(
        batch_size, num_views, 3, img_height, img_width,
        device=points[0].device, dtype=points[0].dtype
    )
    
    for i in range(batch_size):
        cur_feat = img_feat[i].transpose(0, 1)  
        cur_pos = pixel_positions[i].long()  # [n_points, 3] (view, y, x)
        
        restored[i, 
                 cur_pos[:, 0],  # view index
                 :,              # all channels
                 cur_pos[:, 1],  # y position
                 cur_pos[:, 2]   # x position
                ] = cur_feat.transpose(0, 1)
        if points[i] is not None:
            cur_points = points[i].transpose(0, 1)  
            restored_points[i, cur_pos[:, 0], :, cur_pos[:, 1], cur_pos[:, 2]] = cur_points.transpose(0, 1)

    return restored, restored_points


@torch.no_grad()
def get_pt_loc_on_img(points_3d, intrinsics, extrinsics):
    """
    Project 3D points onto image planes of the RLBench views.

    Params:
        points_3d (torch.Tensor): [bs, np, 3]
        intrinsics (List[torch.Tensor]): [bs, 3, 3]  len = 4
        extrinsics (List[torch.Tensor]): [bs, 4, 4]  len = 4

    Return:
        torch.Tensor: [bs, 1, 4, 2] 2D pixel locations on each view
    """
    
    assert len(points_3d.shape) == 3 and points_3d.shape[-1] == 3
    bs = points_3d.shape[0]
    num_cameras = len(intrinsics)

    all_points_2d = []
    for j in range(bs):
        sample_points_2d = []
        for i in range(num_cameras):
            
            K = intrinsics[i][j]  # [3, 3]
            E = extrinsics[i][j]  # [4, 4]
            R = E[:3, :3]                   # [3, 3]
            t = E[:3, 3].unsqueeze(-1)      # [3, 1]
            Rt = torch.cat([R, t], dim=1)   # [3, 4]
            P = K @ Rt                      # [3, 4]
            
            point_3d = points_3d[j].squeeze(0)         # [3]
            point_homogeneous = torch.cat([point_3d, torch.tensor([1.0], device = point_3d.device)])  # [4]
            projected = P @ point_homogeneous  # [3]

            z = projected[2].clamp(min=1e-6)
            x = projected[0] / z
            y = projected[1] / z
            sample_points_2d.append(torch.stack([x, y]))    # [2]
            
        sample_points_2d = torch.stack(sample_points_2d)    # [4, 2]
        all_points_2d.append(sample_points_2d.unsqueeze(0)) # [1, 4, 2]
        
    result = torch.cat(all_points_2d, dim=0)
    # [bs, 1, 4, 2]
    return result.unsqueeze(1)


@torch.no_grad()
def get_max_3d_frm_hm_cube(hm, intrinsics, extrinsics, topk=1, non_max_sup=False, 
                           non_max_sup_dist=0.02):
    """
    only for eval
    given set of heat maps, return the 3d location of the point with the
        largest score, assumes the points are in a cube [-1, 1]. This function
        should be used  along with the intrinsics and extrinsics of cameras. 
    :param hm: (1, nc, h, w)
    :return: (1, topk, 3)
    """

    x, nc, h, w = hm.shape
    assert x == 1
    pts_hm, pts = get_feat_frm_hm_cube(hm, intrinsics, extrinsics) 
    # (bs, np, nc)
    pts_hm = pts_hm.permute(2, 1, 0)
    # (bs, np)
    pts_hm = torch.mean(pts_hm, -1)

    if non_max_sup and topk > 1:
        _pts = pts.clone()
        pts = []
        pts_hm = torch.squeeze(pts_hm, 0)
        for i in range(topk):
            ind_max_pts = torch.argmax(pts_hm, -1)
            sel_pts = _pts[ind_max_pts]
            pts.append(sel_pts)
            dist = torch.sqrt(torch.sum((_pts - sel_pts) ** 2, -1))
            pts_hm[dist < non_max_sup_dist] = -1
        pts = torch.stack(pts, 0).unsqueeze(0)
    else:
        # (bs, topk)
        ind_max_pts = torch.topk(pts_hm, topk)[1]
        # (bs, topk, 3)
        pts = pts[ind_max_pts]
    return pts

 
@torch.no_grad()
def get_feat_frm_hm_cube(hm, intrinsics, extrinsics):
    """
    only for eval
    :param hm: torch.Tensor of (1, num_img, h, w)
    :return: tupe of ((num_img, h^3, 1), (h^3, 3))
    """
    x, nc, h, w = hm.shape
    assert x == 1
    # assert nc == 4
    split_tensors = intrinsics.unbind(dim=0)
    intrinsics = [t.unsqueeze(0) for t in split_tensors]
    split_tensors_e = extrinsics.unbind(dim=0)
    extrinsics = [t.unsqueeze(0) for t in split_tensors_e]

    res = h
    pts = torch.linspace(-1 + (1 / res), 1 - (1 / res), res, device = hm.device)
    pts = torch.cartesian_prod(pts, pts, pts)

    P_list = []
    for i in range(nc):
        K = intrinsics[i]  
        E = extrinsics[i]  
        R = E[0, :3, :3]   
        t = E[0, :3, 3].unsqueeze(-1)  
        Rt = torch.cat([R, t], dim=1)  
        P = K[0] @ Rt    
        P_list.append(P)

    point_homogeneous = torch.cat([
        pts,
        torch.ones(pts.shape[0], 1, device=pts.device)
    ], dim=1)  

    pts_img = []
    for P in P_list:
        projected = point_homogeneous @ P.T  
        z = projected[:, 2].clamp(min=1e-6)
        x = projected[:, 0] / z
        y = projected[:, 1] / z
        pts_img.append(torch.stack([x, y], dim=1))  

    pts_img = torch.stack(pts_img, dim=1)  
    pts_img = pts_img.permute(1, 0, 2)      # [4, res^3, 2]

    fix_pts_hm, pts_cam, pts_cam_wei = select_feat_from_hm(
        pts_img, hm.transpose(0, 1)[0 : nc]
    )
    # fix_pts_hm = select_feat_from_hm_cache(
    #     pts_cam, hm.transpose(0, 1)[0 : nc], pts_cam_wei
    # )
    
    return fix_pts_hm, pts


class DPTHead_Custom(nn.Module):
    """
    Custom DPT Head for image encoding.

    """
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super(DPTHead_Custom, self).__init__()
        assert pos_embed == False
        assert feature_only == True
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features

        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_preds_16 = []
        all_preds_32 = []
        all_preds_64 = []
        all_preds_128 = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            chunk_output, out_16, out_32, out_64, out_128 = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
            all_preds_16.append(out_16)
            all_preds_32.append(out_32)
            all_preds_64.append(out_64)
            all_preds_128.append(out_128)

        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1), torch.cat(all_preds_16, dim=1), torch.cat(all_preds_32, dim=1), \
            torch.cat(all_preds_64, dim=1), torch.cat(all_preds_128, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)

            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out, out_16, out_32, out_64, out_128 = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        out_16 = out_16.view(B, S, *out_16.shape[1:])
        out_32 = out_32.view(B, S, *out_32.shape[1:]) 
        out_64 = out_64.view(B, S, *out_64.shape[1:]) 
        out_128 = out_128.view(B, S, *out_128.shape[1:])
        return out.view(B, S, *out.shape[1:]), out_16, out_32, out_64, out_128
    
    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out_16 = out # (32, 128, 16, 16)

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out_32 = out # (32, 128, 32, 32)

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out_64 = out # (32, 128, 64, 64)

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out_128 = out # (32, 128, 128, 128)

        out = self.scratch.output_conv1(out)
        return out, out_16, out_32, out_64, out_128


def visualize_depth(rgb_img, depth, depth_conf, epoch, iteration, save_dir="depth_visualization"):
    assert rgb_img.shape[0] == depth.shape[0] == depth_conf.shape[0]
    assert rgb_img.shape[1] == depth.shape[1] == depth_conf.shape[1]
    assert rgb_img.shape[3] == depth.shape[2] == depth_conf.shape[2]
    assert rgb_img.shape[4] == depth.shape[3] == depth_conf.shape[3]
    os.makedirs(save_dir, exist_ok=True)
    iter_save_dir = os.path.join(save_dir, f"ep{epoch}_iter_{iteration}")
    os.makedirs(iter_save_dir, exist_ok=True)

    bs, num_view = rgb_img.shape[0], rgb_img.shape[1]
    depth_flat = depth.permute(0, 1, 4, 2, 3)
    depth_conf_flat = depth_conf.unsqueeze(2)

    for i in range(bs):
        for j in range(num_view):
            fig, ax = plt.subplots()
            im = ax.imshow(depth_flat[i, j].squeeze().detach().cpu().numpy(), cmap='viridis')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(os.path.join(iter_save_dir, f"predict_ep{epoch}_iter{iteration}_sample{i}_view{j}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

    for i in range(bs):
        for j in range(num_view):
            fig, ax = plt.subplots()
            im = ax.imshow(depth_conf_flat[i, j].squeeze().detach().cpu().numpy(), cmap='hot')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(os.path.join(iter_save_dir, f"confidence_ep{epoch}_iter{iteration}_sample{i}_view{j}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

    for i in range(bs):
        for j in range(num_view):
            img = rgb_img[i, j].detach().cpu().permute(1, 2, 0).numpy()
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(os.path.join(iter_save_dir, f"rgb_ep{epoch}_iter{iteration}_sample{i}_view{j}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def extract_vggt_features(rgb_vggt, model, device, return_attn=False):
     
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=vggt_dtype):
            images = rgb_vggt  # add batch dimension
            if return_attn:
                aggregated_tokens_list, ps_idx, attn = model.aggregator(images, return_attn=True)  # attn (B*S, num_heads, P, P) 全局注意力权重矩阵
            else:
                aggregated_tokens_list, ps_idx = model.aggregator(images, return_attn=False)
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_maps_by_unprojection = []
        for i in range(depth_map.size(0)):  
            point_map_by_unprojection = unproject_depth_map_to_point_map(
                depth_map[i].cpu().numpy(), # (V, 518, 518, 1)
                extrinsic[i].cpu().numpy(), # (V, 3, 4)
                intrinsic[i].cpu().numpy()
            )
            point_maps_by_unprojection.append(torch.from_numpy(point_map_by_unprojection).float())
        point_map_by_unprojection = torch.stack(point_maps_by_unprojection) # (B, V, 518, 518, 3)

        # point_map_view_1, point_map_view_2 = point_map[0, 0], point_map[0, 1]
        point_map_view_1 = point_map_by_unprojection[:,0,...].detach().clone().to(device)
        point_map_view_2 = point_map_by_unprojection[:,1,...].detach().clone().to(device)
        point_map_view_3 = point_map_by_unprojection[:,2,...].detach().clone().to(device)
        point_conf_view_1, point_conf_view_2, point_conf_view_3 = point_conf[:, 0], point_conf[:, 1], point_conf[:, 2]
        extrinsic_1, extrinsic_2, extrinsic_3 = extrinsic[:, 0], extrinsic[:, 1], extrinsic[:, 2]
        intrinsic_1, intrinsic_2, intrinsic_3 = intrinsic[:, 0], intrinsic[:, 1], intrinsic[:, 2]
        depth_pred_1, depth_pred_2 = depth_map[:, 0].squeeze(-1), depth_map[:, 1].squeeze(-1)
        depth_pred_3 = depth_map[:, 2].squeeze(-1)

        image_shape = tuple(rgb_vggt.shape[-2:])
        
        if return_attn:
            # 拆分为4个 (1, num_heads, P, P) 的自注意力权重 （14，23, 32, 41）
            cost_1, cost_2, cost_3, cost_4 = attn.chunk(4, dim=0)   
            cost_1 = cost_1.mean(dim=1)             # 多头注意力关注不同特征，取平均得到更鲁棒的相似度矩阵 (B, P, P)
            cost_2 = cost_2.mean(dim=1)
            cost_3 = cost_3.mean(dim=1)
            # cost_4 = cost_4.mean(dim=1)
            return {
                'point_map_view_1': point_map_view_1,   # (B, 518, 518, 3)
                'point_map_view_2': point_map_view_2,
                'point_map_view_3': point_map_view_3,
                # 'point_map_view_4': point_map_view_4,
                'point_conf_view_1': point_conf_view_1, # (B, 518, 518)
                'point_conf_view_2': point_conf_view_2,
                'point_conf_view_3': point_conf_view_3, 
                # 'point_conf_view_4': point_conf_view_4,
                'extrinsic_1': extrinsic_1,             # (B, 3, 4)
                'extrinsic_2': extrinsic_2,
                'extrinsic_3': extrinsic_3,
                # 'extrinsic_4': extrinsic_4,
                'intrinsic_1': intrinsic_1,             # (B, 3, 3)
                'intrinsic_2': intrinsic_2,
                'intrinsic_3': intrinsic_3,
                # 'intrinsic_4': intrinsic_4,
                'depth_pred_1': depth_pred_1,           # (B, 518, 518)
                'depth_pred_2': depth_pred_2,
                'depth_pred_3': depth_pred_3,
                # 'depth_pred_4': depth_pred_4,
                'image_shape': image_shape,
                'cost_1': cost_1,                       # (B, P, P)
                'cost_2': cost_2,
                'cost_3': cost_3,
                # 'cost_4': cost_4,
                # 'aggregated_tokens_list': aggregated_tokens_list,
                'images': images,                       # (B, V, 3, 518, 518)
                'ps_idx': ps_idx                        # 5
            }, aggregated_tokens_list

    return {
        'point_map_view_1': point_map_view_1,   # (B, 518, 518, 3)
        'point_map_view_2': point_map_view_2,
        'point_map_view_3': point_map_view_3,
        'point_conf_view_1': point_conf_view_1, # (B, 518, 518)
        # 'point_conf_view_2': point_conf_view_2,
        # 'point_conf_view_3': point_conf_view_3, 
        # 'point_conf_view_4': point_conf_view_4,
        'extrinsic_1': extrinsic_1,             # (B, 3, 4)
        'extrinsic_2': extrinsic_2,
        'extrinsic_3': extrinsic_3,
        # 'extrinsic_4': extrinsic_4,
        'intrinsic_1': intrinsic_1,             # (B, 3, 3)
        'intrinsic_2': intrinsic_2,
        'intrinsic_3': intrinsic_3,
        # 'intrinsic_4': intrinsic_4,
        'depth_pred_1': depth_pred_1,           # (B, 518, 518)
        'depth_pred_2': depth_pred_2,
        'depth_pred_3': depth_pred_3,
        # 'depth_pred_4': depth_pred_4
    }, image_shape, images, aggregated_tokens_list, ps_idx


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


def sample_keypoints_nms(mask, conf, N, min_distance, device=None):
    if device is None:
        device = mask.device
    B, H, W = mask.shape

    score_map = torch.zeros_like(mask, dtype=torch.float32, device=device)
    score_map[mask] = conf[mask]
    
    kernel_size = int(min_distance) * 2 + 1
    pad = kernel_size // 2

    pooled = F.max_pool2d(score_map.unsqueeze(1),
                          kernel_size=kernel_size,
                          stride=1,
                          padding=pad).squeeze(1) 

    eps = 1e-6
    nms_mask = (score_map - pooled).abs() < eps
    nms_mask = nms_mask & mask
    keypoints_list = []
    for b in range(B):
        keypoints = torch.nonzero(nms_mask[b], as_tuple=False)  # (M, 2)
        M = keypoints.shape[0]
        if M == 0:
            print("No keypoints found by nms.")
            keypoints_list.append(torch.zeros((N, 2), device=device, dtype=torch.int64))
        elif M > N:
            perm = torch.randperm(M, device=device)[:N]
            sampled_keypoints = keypoints[perm]
            keypoints_list.append(sampled_keypoints)
        else:
            # 如果关键点不足 N 个，重复采样
            repeat_times = (N + M - 1) // M
            sampled_keypoints = torch.repeat_interleave(keypoints, repeat_times, dim=0)[:N]
            keypoints_list.append(sampled_keypoints)
    return torch.stack(keypoints_list)  # (B, N, 2)


def compute_projection(P, points_3d):
    """    
    Args:
        P: (B, 3, 4) torch tensor, projection matrix.
        points_3d: (B, ..., 3) tensor of 3D world points.
        
    Returns:
        proj_points: (B, ..., 2) tensor of 2D pixel coordinates.
    """
    B = P.shape[0]
    orig_shape = points_3d.shape[:-1]
    points_flat = points_3d.view(B, -1, 3)  # (B, N, 3)
    ones = torch.ones((B, points_flat.shape[1], 1), device=points_flat.device)
    points_h = torch.cat([points_flat, ones], dim=-1)  # (B, N, 4)

    # Batch matrix multiplication: (B, 3, 4) @ (B, 4, N) -> (B, 3, N)
    proj_h = torch.bmm(P, points_h.transpose(1, 2))

    # Normalize: (B, 3, N) -> (B, N, 3)
    proj_h = proj_h.transpose(1, 2)
    proj_points = proj_h[..., :2] / (proj_h[..., 2:3] + 1e-8)

    # Reshape back to original
    return proj_points.view(*orig_shape, 2)

def get_coview_mask(point_map, P, image_shape):
    """
    Args:
        point_map: (B, H, W, 3)
        P: (B, 3, 3) - projection matrix (intrinsic @ extrinsic[:3])
        image_shape: (H_img, W_img)
    Returns:
        mask: (B, H, W) - valid projection mask
    """
    H_img, W_img = image_shape
    B, H, W, _ = point_map.shape

    proj_points = compute_projection(P, point_map)  # (B, H, W, 2)

    u = proj_points[..., 0]
    v = proj_points[..., 1]

    mask = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
    return mask

def convert_camera_to_world(point_map, extrinsic):
    """
    Args:
        point_map: (B, H, W, 3)
        extrinsic: (B, 3, 4) - [R | t]
    Returns:
        world_points: (B, H, W, 3)
    """
    R = extrinsic[:, :, :3]  # (B, 3, 3)
    t = extrinsic[:, :, 3].unsqueeze(1)  # (B, 1, 3)
    R_inv = torch.inverse(R)  # (B, 3, 3)

    # Reshape point_map for batched matmul: (B, H*W, 3)
    B, H, W, _ = point_map.shape
    points_flat = point_map.view(B, -1, 3)  # (B, H*W, 3)

    # Transform: (B, H*W, 3) → (B, 3, H*W)
    transformed = torch.bmm(R_inv, (points_flat - t).transpose(1, 2)).transpose(1, 2)

    return transformed.view(B, H, W, 3)


def get_coview_masks(vggt_features, image_shape):
    """
    Args:
        vggt_features: dict with keys 'point_map_view_1', ..., 'intrinsic_1', ..., 'extrinsic_1', ...
        image_shape: (H, W)
    Returns:
        masks: tuple of (B, H, W) masks for each view
    """
    B = vggt_features['point_map_view_1'].shape

    point_maps = [vggt_features[f'point_map_view_{i}'] for i in range(1, 4)]  # list of (B, H, W, 3)
    extrinsics = [vggt_features[f'extrinsic_{i}'] for i in range(1, 4)]        # list of (B, 3, 4)
    intrinsics = [vggt_features[f'intrinsic_{i}'] for i in range(1, 4)]       # list of (B, 3, 3)

    world_point_maps = []
    for i in range(3):
        world_points = convert_camera_to_world(point_maps[i], extrinsics[0])  # (B, H, W, 3)
        world_point_maps.append(world_points)

    Ps = [torch.bmm(intrinsics[i], extrinsics[i]) for i in range(3)]  # (B, 3, 3)

    pairings = [(0, 1), (1, 2), (2, 0)]  # view1 ↔ view4, view2 ↔ view3 等

    masks = []
    for src, dst in pairings:
        P = Ps[dst]
        world_points = world_point_maps[src]
        mask = get_coview_mask(world_points, P, image_shape)  # (B, H, W)
        masks.append(mask)

    return tuple(masks)


def sample_keypoints(vggt_features, image_shape, images, aggregated_tokens_list, ps_idx, model, device, num_keypoints=300, min_distance=5):

    point_conf_view_1 = vggt_features['point_conf_view_1']
    
    mask_1, mask_2, mask_3 = get_coview_masks(vggt_features, image_shape) # (B, H, W)
    
    # 在mask为True的有效区域内，通过非极大值抑制（NMS）筛选出置信度图conf的局部最大值，最终返回最多300个关键点的坐标 (B, 300, 2)
    sampled_kp_1 = sample_keypoints_nms(mask_1, point_conf_view_1, N=num_keypoints, min_distance=min_distance, device=device)

    if sampled_kp_1 is None:
        print("No keypoints found in the first view.")
        return None, None, None, None, None
    sampled_kp_1 = sampled_kp_1[:, :, [1, 0]].int()  # (row, col) -> (x, y)
    # list of length 4 (B, V, 2, 2)
    sampled_kp_o, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=sampled_kp_1)

    sampled_kp_2 = sampled_kp_o[-1][:, 1].int()  # (x, y)
    sampled_kp_3 = sampled_kp_o[-1][:, 2].int()
    # sampled_kp_4 = sampled_kp_o[-1][:, 3].int()
    
    mh, mw = image_shape
    valid_kp_1 = (sampled_kp_1[:, :, 0] >= 3) & (sampled_kp_1[:, :, 0] < int(mw) - 3) & (sampled_kp_1[:, :, 1] >= 3) & \
        (sampled_kp_1[:, :, 1] < int(mh) - 3)
    valid_kp_2 = (sampled_kp_2[:, :, 0] >= 3) & (sampled_kp_2[:, :, 0] < int(mw) - 3) & (sampled_kp_2[:, :, 1] >= 3) & \
        (sampled_kp_2[:, :, 1] < int(mh) - 3)
    valid_kp_3 = (sampled_kp_3[:, :, 0] >= 3) & (sampled_kp_3[:, :, 0] < int(mw) - 3) & (sampled_kp_3[:, :, 1] >= 3) & \
        (sampled_kp_3[:, :, 1] < int(mh) - 3)
    # valid_kp_4 = (sampled_kp_4[:, :, 0] >= 3) & (sampled_kp_4[:, :, 0] < int(mw) - 3) & (sampled_kp_4[:, :, 1] >= 3) & \
    #     (sampled_kp_4[:, :, 1] < int(mh) - 3)
    valid_kp = valid_kp_1 & valid_kp_2 & valid_kp_3  # (B, 300)

    kp_1, kp_2, kp_3 = [], [], []           # list of length B 
    for b in range(valid_kp.shape[0]):
        mask_b = valid_kp[b]                # (300,)
        kp_b_1 = sampled_kp_1[b][mask_b]    # (N_b, 2)
        kp_b_2 = sampled_kp_2[b][mask_b]
        kp_b_3 = sampled_kp_3[b][mask_b]
        # kp_b_4 = sampled_kp_4[b][mask_b]
        kp_1.append(kp_b_1)
        kp_2.append(kp_b_2)
        kp_3.append(kp_b_3)
        # kp_4.append(kp_b_4)
    
    return kp_1, kp_2, kp_3, valid_kp, mask_1, mask_2, mask_3


def get_3d_preprocess(replay_sample, vggt_model, device):
    rgb_vggt_list = []
    rgb_front_list = []
    rgb_left_list = []
    rgb_right_list = []
    rgb_wrist_list = []
    for j in range(len(replay_sample["episode_idx"])):  
        sample = {k: v[j].cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in replay_sample.items()}
        
        index = sample["episode_idx"] # 0 to 99
        if sample["keypoint_frame"] != -1:
            i = sample["keypoint_frame"] # or keypoint_frame
        else:
            i = sample["sample_frame"] # or keypoint_frame
        sample_task = replay_sample["tasks"][j]
        data_path = os.path.join(DATA_FOLDER, f"rlbench/train/{sample_task}/all_variations/episodes")
        episode_path = os.path.join(data_path, EPISODE_FOLDER % index)

        # resize for VGGT
        img_front_path = os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)
        img_left_path = os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)
        img_right_path = os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)
        img_wrist_path = os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)

        rgb_front = cv2.imread(str(img_front_path))[..., ::-1].copy()
        # cv2.imwrite(f"debug_runs/images/temp/output{j}_front.jpg", rgb_front) 
        rgb_front = np.moveaxis((rgb_front / 255.).astype(np.float32), -1, 0)
        rgb_left = cv2.imread(str(img_left_path))[..., ::-1].copy()
        # cv2.imwrite(f"debug_runs/images/temp/output{j}_left.jpg", rgb_left) 
        rgb_left = np.moveaxis((rgb_left / 255.).astype(np.float32), -1, 0)
        rgb_right = cv2.imread(str(img_right_path))[..., ::-1].copy()
        # cv2.imwrite(f"debug_runs/images/temp/output{j}_right.jpg", rgb_right) 
        rgb_right = np.moveaxis((rgb_right / 255.).astype(np.float32), -1, 0)
        rgb_wrist = cv2.imread(str(img_wrist_path))[..., ::-1].copy()
        # cv2.imwrite(f"debug_runs/images/temp/output{j}_wrist.jpg", rgb_wrist) 
        rgb_wrist = np.moveaxis((rgb_wrist / 255.).astype(np.float32), -1, 0)

        rgb_vggt = load_and_preprocess_images([str(img_front_path), str(img_left_path), 
                                               str(img_right_path), str(img_wrist_path)])
        rgb_vggt_list.append(rgb_vggt)
        rgb_front_list.append(rgb_front)
        rgb_left_list.append(rgb_left)
        rgb_right_list.append(rgb_right)
        rgb_wrist_list.append(rgb_wrist)

    rgb_vggt = torch.stack(rgb_vggt_list)  # (B, V, C, H, W)
    vggt_features, image_shape, images, aggregated_tokens_list, ps_idx = extract_vggt_features(rgb_vggt.to(device), vggt_model, device=device, 
                                                                                               return_attn=False)

    (kp_1, kp_2, kp_3, valid_kp, mask_1, mask_2, mask_3) = sample_keypoints(vggt_features, image_shape, images, aggregated_tokens_list, ps_idx, 
                                                                            vggt_model, device=device, num_keypoints=300, min_distance=5)

    mh, mw = image_shape
    # (B, C=3, H, W)
    rgb_front_resized = F.interpolate(torch.from_numpy(np.array(rgb_front_list)).float().to(device), size=(mh, mw))#, mode='bicubic', align_corners=False
    rgb_left_resized = F.interpolate(torch.from_numpy(np.array(rgb_left_list)).float().to(device), size=(mh, mw))
    rgb_right_resized = F.interpolate(torch.from_numpy(np.array(rgb_right_list)).float().to(device), size=(mh, mw))
    rgb_wrist_resized = F.interpolate(torch.from_numpy(np.array(rgb_wrist_list)).float().to(device), size=(mh, mw))
    
    return vggt_features, kp_1, kp_2, kp_3, mask_1, mask_2, mask_3, valid_kp, rgb_front_resized, rgb_left_resized, rgb_right_resized, rgb_wrist_resized


def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid

    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def get_eval_parser():
    parser = argparse.ArgumentParser()
    # parser.set_defaults(entry=lambda cmd_args: parser.print_help()) # extra for train
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["all_rlbench"]
    )
    parser.add_argument("--model-folder", type=str, 
                        default="/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench/vggtact_debug_rlbench/stage2")
    parser.add_argument("--eval-datafolder", type=str, 
                        default="/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench/test/")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--vggt_cfg_path", type=str, default=None)

    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--log-name", type=str, default='')
    parser.add_argument("--log-dir", type=str, default="evals")
    parser.add_argument("--model-name", type=str, default="model_24.pth")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)

    return parser
    

def save_rgb_images(images, base_path, prefix="", views=None):
    os.makedirs(base_path, exist_ok=True)
    batch_size, num_views, _, height, width = images.shape
    views = views or [f"view_{i}" for i in range(num_views)]
    for b in range(batch_size):
        for v in range(num_views):
            img_np = images[b, v].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            filename = f"{prefix}_b{b}_{views[v]}.png"
            cv2.imwrite(os.path.join(base_path, filename), img_bgr)
    

def save_depth_images(pc_img, save_dir, epoch, iteration, prefix="depth", views=None):
    os.makedirs(save_dir, exist_ok=True)
    iter_save_dir = os.path.join(save_dir, f"ep{epoch}_iter_{iteration}")
    os.makedirs(iter_save_dir, exist_ok=True)

    depth_maps = pc_img.unsqueeze(2)
    depth_maps = depth_maps.permute(0, 1, 3, 4, 2)
    depth_maps = depth_maps.cpu().numpy() 

    # depth_maps frm pc: (8, 4, 128, 128, 1) -0.8608351 ~ 1.0617968       
    depth_maps = depth_maps.squeeze(-1)
    # print("[DEBUG] depth check: ", depth_maps.min(), depth_maps.max())
    for i in range(depth_maps.shape[0]):
        for v in range(depth_maps.shape[1]):
            depth = depth_maps[i, v]  
            fig, ax = plt.subplots()
            im = ax.imshow(depth, cmap='viridis')  
            ax.axis('off')  
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            save_path = os.path.join(iter_save_dir, f"{prefix}_sample{i}_view{views[v]}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()


def get_depth(pixel_positions, points):
    batch_size = len(pixel_positions)
    num_views = len(CAMERAS)
    img_height = IMAGE_SIZE
    img_width = IMAGE_SIZE
    
    restored_points = torch.full(
        size=(batch_size, num_views, 3, img_height, img_width),
        fill_value=-2,
        device=points[0].device,
        dtype=points[0].dtype
    )
    
    for i in range(batch_size):
        cur_pos = pixel_positions[i].long()  # [n_points, 3] (view, y, x)       
        if points[i] is not None:
            cur_points = points[i].transpose(0, 1)  
            restored_points[i, cur_pos[:, 0], :, cur_pos[:, 1], cur_pos[:, 2]] = cur_points.transpose(0, 1)

    depth_maps = restored_points[:, :, 2:3].clone()
    invalid_mask = (depth_maps == -2)
    valid_mask = ~invalid_mask

    sum = torch.sum(depth_maps, (0, 1)) + torch.sum(invalid_mask, (0, 1)) # (1, 128, 128)
    mean = sum / ((img_height * img_width) - torch.sum(invalid_mask, (0, 1)))
    
    mean = mean.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
    depth_maps -= mean.to(depth_maps.device)
    depth_maps[invalid_mask] = -1.5  
    return depth_maps, valid_mask

def get_depth_st2(rendered_img):
    batch_size = rendered_img.shape[0]
    num_views = rendered_img.shape[1]
    img_height = rendered_img.shape[3]
    img_width = rendered_img.shape[4]

    depth_maps = rendered_img[:, :, 6].view(batch_size, num_views, 1, img_height, img_width)
    invalid_mask = (depth_maps == -10)
    valid_mask = ~invalid_mask

    return depth_maps, valid_mask




def resize_gt_depth(depth_maps, valid_mask, size):
    bs, view, depth_dim, h, w = depth_maps.shape 
    gt_depth = depth_maps.permute(0, 1, 3, 4, 2)
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth_flat = gt_depth.view(bs * view, depth_dim, h, w)                
    valid_mask_flat = valid_mask.view(-1, *valid_mask.shape[2:])  # [b*v, 1, 128, 128]

    gt_depth_resized = F.interpolate(
        gt_depth_flat, 
        size=(size, size), 
        mode='bicubic', 
        align_corners=False
    )  # [B*V, 1, 224, 224]
    valid_mask_resized = F.interpolate(
        valid_mask_flat.float(), 
        size=(size, size), 
        mode='nearest'  # 对掩码使用最近邻插值（避免引入浮点值）
    ).bool()  # [B*V, 1, 224, 224]

    gt_depth_resized = gt_depth_resized.view(bs, view, size, size, depth_dim)
    valid_mask_resized = valid_mask_resized.view(bs, view, size, size)
    return gt_depth_resized, valid_mask_resized

def visualize_point_cloud(pc, title="Point Cloud", save_path=None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='blue')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()




# loss tools
def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """
        
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, 
              affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False, postfix=""):
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    # if affine_inv:
    #     scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
    #     pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = \
        reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss)


    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - \
            alpha * torch.log(first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[other_frames_mask] - \
            alpha * torch.log(other_frames_conf[other_frames_mask])


    if conf_loss_first_frame.numel() >0 and conf_loss_other_frames.numel() >0:
        assert not valid_range>0
        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss")


    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames

    # Verified that the loss is the same
    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


from scipy.optimize import least_squares
def triangulate_point(keypoints_2d, camera_matrices):
    """通过多视角2D点 + 相机矩阵三角化3D坐标
    Args:
        keypoints_2d: List of [x, y] in each view (M, 2)
        camera_matrices: List of [3x4] projection matrices (M, 3, 4)
    Returns:
        3D point (3,)
    """
    def residuals(X, points, matrices):
        proj = np.array([m @ np.hstack([X, 1]).reshape(4, 1) for m in matrices])
        proj = (proj / proj[:, 2]).squeeze(axis=-1)[:, :2]
        return (proj - points).ravel()

    # 初始猜测（原点）
    x0 = np.zeros(3)
    result = least_squares(residuals, x0, args=(keypoints_2d, camera_matrices))
    return result.x


def get_3d_keypoints(data_test):
    """计算900个关键点的3D坐标
    Args:
        data_test: 包含以下键的字典:
            - kp_1/kp_2/kp_3: 三个视角的2D关键点 (300, 2)
    Returns:
        torch.Tensor: 3D坐标 (900, 3)
    """
    # 1. 定义三个视角的投影矩阵
    K = np.array([
        [[112., 0., 112.], [0., 112., 112.], [0., 0., 1.]],  # front
        [[112., 0., 112.], [0., 112., 112.], [0., 0., 1.]],  # top
        [[112., 0., 112.], [0., 112., 112.], [0., 0., 1.]]   # right
    ])
    RT = np.array([
        [[1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., -1., 1.]],      # front
        [[0., 0., -1., 1.], [1., 0., 0., 0.], [0., -1., 0., 0.]],      # top
        [[-1., 0., 0., 0.], [0., 0., -1., 0.5], [0., -1., 0., 0.]]     # right
    ])
    projection_matrices = np.matmul(K, RT)  # (3, 3, 4)

    # 2. 获取所有2D关键点并合并 (900, 2)
    kp_all = np.concatenate([
        data_test['kp_1'].cpu().numpy() if isinstance(data_test['kp_1'], torch.Tensor) else data_test['kp_1'],
        data_test['kp_2'].cpu().numpy() if isinstance(data_test['kp_2'], torch.Tensor) else data_test['kp_2'],
        data_test['kp_3'].cpu().numpy() if isinstance(data_test['kp_3'], torch.Tensor) else data_test['kp_3']
    ], axis=0)

    # 3. 三角化所有点（假设每组的三个视角点按顺序排列）
    kp_3d = np.zeros((900, 3), dtype=np.float32)
    for i in range(300):  # 每组3个点（front/top/right各1个）
        idx = [i, i+300, i+600]  # 对应三个视角中的同一点
        points_2d = kp_all[idx]   # (3, 2)
        
        # 三角化
        kp_3d[i] = triangulate_point(points_2d, projection_matrices)

    return torch.from_numpy(kp_3d)  # (900, 3)



def get_3d_keypoints_from_gt(data_test, view_names=['front', 'top', 'right']):  # 'front', 'left_shoulder', 'right_shoulder', 'wrist'
    view_to_idx = {'front': 0, 'top': 1, 'right': 2} # 视角名称到相机索引的映射  
    kp_3d_gt = {}
    scale = 128 / 518  # kp输入是 518x518 图像输出为 128x128
    batch_size = 1#next(v for v in data_test.values() if isinstance(v, (torch.Tensor, np.ndarray))).shape[0]
    
    # 准备相机投影矩阵 P = K @ [R|t] (M, 3, 4)
    K = [[[112.,   0., 112.],
         [  0., 112., 112.],
         [  0.,   0.,   1.]],

        [[112.,   0., 112.],
         [  0., 112., 112.],
         [  0.,   0.,   1.]],

        [[112.,   0., 112.],
         [  0., 112., 112.],
         [  0.,   0.,   1.]]]   # (M, 3, 3)
    
    RT = [[[ 1.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000, -1.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000, -1.0000,  1.0000]],

        [[ 0.0000,  0.0000, -1.0000,  1.0000],
         [ 1.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000, -1.0000,  0.0000,  0.0000]],

        [[-1.0000,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000, -1.0000,  0.5000],
         [ 0.0000, -1.0000,  0.0000,  0.0000]]]    # (M, 3, 4)
    
    projection_matrices = np.matmul(K, RT)  # (M, 3, 4)

    for idx, view in enumerate(view_names, 1):
        kp_data = data_test[f'kp_{idx}'].squeeze()
        
        if isinstance(kp_data, torch.Tensor):
            kp_2d = kp_data.cpu().numpy().astype(np.float32)  # (B, N, 2)
        else:
            kp_2d = kp_data.astype(np.float32)
        kp_pixel = (kp_2d * scale).astype(int)              # (300, 2)
        kp_pixel = np.clip(kp_pixel, [0, 0], [127, 127])    # 128x128 图像
        print("DEBUG batch_size = ", batch_size)
        # # 获取点云数据 (B, 3, W, H)
        # point_cloud = data_test[f'{view}_point_cloud'].squeeze(1) # shape: (3, W, H) or (B, 1, 3, W, H)
        # if isinstance(point_cloud, torch.Tensor):
        #     point_cloud = point_cloud.cpu().numpy()
        
        # 初始化输出数组 (B, N, 3)
        batch_kp_3d = np.zeros((batch_size, kp_pixel.shape[1], 3), dtype=np.float32)
        # for b in range(batch_size):
        #     # 处理当前batch样本
        #     # pc = point_cloud[b]  # (3, W, H)
        #     # W, H = pc.shape[1], pc.shape[2]
            
        #     for n in range(kp_pixel.shape[1]):
        #         x, y = kp_pixel[b, n]
        #         if 0 <= x < W and 0 <= y < H:
        #             xyz_world = pc[:, x, y]
        #             if not np.all(xyz_world == 0):  # 过滤零值点
        #                 batch_kp_3d[b, n] = xyz_world
        
        # kp_3d_gt[f'kp_{idx}'] = batch_kp_3d  
        for b in range(batch_size):
            for n in range(kp_pixel.shape[1]):
                # 收集当前关键点在所有视角的2D坐标
                keypoints_2d = []
                matrices = []
                
                # 遍历所有原始视角（4个）
                for src_view in ['front', 'left_shoulder', 'right_shoulder', 'wrist']:
                    # 获取当前视角的相机索引
                    cam_idx = view_to_idx[src_view] # only front, top, right
                    
                    # 获取点云坐标 (3, H, W)
                    pc = data_test[f'{src_view}_point_cloud']    
                    if isinstance(pc, torch.Tensor):
                        pc = pc.cpu().numpy()
                    
                    # 投影验证：检查2D点是否在点云中有效
                    x, y = kp_pixel[n]
                    print("DEBUG pc.shape = ", pc.shape)
                    H, W = pc.shape[0], pc.shape[1]
                    if 0 <= x < W and 0 <= y < H and not np.all(pc[:, y, x] == 0):
                        keypoints_2d.append([x, y])
                        matrices.append(projection_matrices[cam_idx])
                
                # 至少需要2个视角才能三角化
                if len(keypoints_2d) >= 2:
                    batch_kp_3d[b, n] = triangulate_point(keypoints_2d, matrices)
        kp_3d_gt[f'kp_{idx}'] = torch.from_numpy(batch_kp_3d)
        # _, _, _, W, H = point_cloud.shape  # 注意这里是W,H不是H,W
        # # print(f"\nView {view} point cloud:")
        # # print(f"Shape: {point_cloud.shape}")
        # # print(f"X range: [{np.min(point_cloud[...,0]):.2f}, {np.max(point_cloud[...,0]):.2f}]")
        # # print(f"Y range: [{np.min(point_cloud[...,1]):.2f}, {np.max(point_cloud[...,1]):.2f}]")
        # # print(f"Z range: [{np.min(point_cloud[...,2]):.2f}, {np.max(point_cloud[...,2]):.2f}]")
        # kp_3d = []
        # valid_count = 0
        # for x, y in kp_pixel: # (B, 1, 1, 300, 2)
        #     if 0 <= x < W and 0 <= y < H:
        #         # 直接获取世界坐标系下的点（无需再转换）
        #         xyz_world = point_cloud[:, x, y]
        #         if not np.all(xyz_world == 0):  # 过滤零值点
        #             kp_3d.append(xyz_world)
        #             valid_count += 1
        #         else:
        #             kp_3d.append([0.0, 0.0, 0.0])  # 保证形状一致
        #     else:
        #         kp_3d.append([0.0, 0.0, 0.0])  # 保证形状一致
        # kp_3d_gt[f'kp_{idx}'] = np.array(kp_3d, dtype=np.float32)
        # # print(f"Extracted {valid_count} valid keypoints (from {len(kp_pixel)} raw points)")
    # for idx in range(1, len(view_names)+1):
    #     kp_3d_gt[f'kp_{idx}'] = torch.from_numpy(kp_3d_gt[f'kp_{idx}']).to(data_test['front_point_cloud'].device)      
    return kp_3d_gt 



def get_common_3d_keypoints(kp_3d_gt, distance_threshold=0.01):
    """
    从四个视角的关键点中找出在三维空间中重合或距离小于阈值的点
    
    参数:
        kp_3d_gt: 字典，包含四个视角的关键点 {'kp_1': array, 'kp_2': array, ...}
        distance_threshold: 判断为同一关键点的最大距离阈值(米)
    
    返回:
        merged_points: 合并后的3D点列表，每个元素为(point, view_indices)
            point: 合并后的3D坐标
            view_indices: 包含该点的视角索引列表(1-4)
        stats: 统计信息字典，包含:
            total_points: 原始总点数
            merged_count: 合并后的点数
            view_coverage: 各共视程度统计
    """
    # 1. 收集所有非零关键点，并记录来源视角
    all_points = []
    for view_idx in range(1, 5):
        points = kp_3d_gt[f'kp_{view_idx}']
        for pt in points:
            if not np.all(pt == 0.0):
                all_points.append((pt, view_idx))
    
    if not all_points:
        return [], {
            'total_points': 0,
            'merged_count': 0,
            'view_coverage': {2:0, 3:0, 4:0}
        }
    
    # 2. 基于距离阈值进行聚类
    merged_clusters = []
    used_indices = set()
    
    # 在聚类时跳过单视角点
    for i, (pt1, view1) in enumerate(all_points):
        if i in used_indices:
            continue
        
        # 仅当找到至少一个邻近点（来自其他视角）时才创建聚类
        cluster = {'points': [pt1], 'views': {view1}}
        has_neighbor = False
        
        for j, (pt2, view2) in enumerate(all_points[i+1:], start=i+1):
            if j in used_indices or view2 == view1:
                continue
                
            dist = np.linalg.norm(pt1 - pt2)
            if dist <= distance_threshold:
                cluster['points'].append(pt2)
                cluster['views'].add(view2)
                used_indices.add(j)
                has_neighbor = True
        
        if has_neighbor:  # 仅保存多视角聚类
            merged_pt = np.mean(cluster['points'], axis=0)
            merged_clusters.append((merged_pt, sorted(cluster['views'])))
    
    # 3. 统计信息
    total_points = sum(len(kp_3d_gt[f'kp_{i}']) for i in range(1, 5))
    
    view_coverage = {2:0, 3:0, 4:0}
    for _, views in merged_clusters:
        num_views = len(views)
        if num_views >= 2:
            view_coverage[min(num_views, 4)] += 1
    
    stats = {
        'total_points': total_points,
        'merged_count': len(merged_clusters),
        'view_coverage': view_coverage
    }
    
    return merged_clusters, stats


def plot_3d_comparison(ax, data_test, kp_3d_pred, view='overview'):
    """3D点云可视化"""
    # 获取ground truth点云（各视角先reshape为(N,3)再合并）
    gt_points = []
    for view_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist']:
        pc = data_test[f'{view_name}_point_cloud']
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        if pc.ndim == 5:    # (B,1,3,H,W)
            pc = pc[0,0]    # 取第一个样本，移除冗余维度
        elif pc.ndim == 4:  # (B,3,H,W)
            pc = pc[0]      # 取第一个样本
        elif pc.ndim == 3:  # (3,W,H)
            pass            
        else:
            raise ValueError(f"Unexpected point cloud shape: {pc.shape}")
            
        pc = pc.transpose(1,2,0)  
        # pc = data_test[f'{view_name}_point_cloud'].transpose(1,2,0)  # shape: (W,H,3)
        gt_points.append(pc.reshape(-1,3))          # (16384, 3)
    gt_points = np.concatenate(gt_points, axis=0)   # (65536, 3)
    # 自动计算合理的坐标范围（保留10%边界）
    x_min, x_max = np.percentile(gt_points[:,0], [5, 95])
    y_min, y_max = np.percentile(gt_points[:,1], [5, 95]) 
    z_min, z_max = np.percentile(gt_points[:,2], [5, 95])
    bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    # 过滤无效点（全零点和NaN）
    valid_mask = ~np.all(gt_points == 0, axis=1) & ~np.isnan(gt_points).any(axis=1)
    gt_points = gt_points[valid_mask]
    # 绘制ground truth点云（使用更高效的绘制方式）
    ax.scatter(gt_points[:,0], gt_points[:,1], gt_points[:,2], 
              c='gray', alpha=0.1, s=5, label='Scene PointCloud')

    # 绘制关键点（过滤无效点）
    colors = ['red', 'green', 'blue', 'yellow']
    for i in range(1, 4):
        kp = kp_3d_pred[f'kp_{i}']  # (300, 3)
        if isinstance(kp, torch.Tensor):
            kp = kp.cpu().numpy()
        # 处理可能的batch维度
        if kp.ndim == 3:  # (B,N,3)
            kp = kp[0]    # 取第一个样本
        valid_kp = kp[~np.all(kp == 0, axis=1)]  # 过滤全零点 
        if len(valid_kp) > 0:
            ax.scatter(valid_kp[0], valid_kp[1], valid_kp[2], #valid_kp[:,0], valid_kp[:,1], valid_kp[:,2],
                       c=colors[i-1], s=30, marker='o', label=f'KP_{i} ({len(valid_kp)} pts)')
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.legend(loc='upper right', fontsize=8)
    # 设置视角和范围
    if view == 'front':
        ax.view_init(elev=0, azim=0)  # 正前方视角
    elif view == 'top_down':
        ax.view_init(elev=90, azim=0)  # 正上方视角
    else:
        ax.view_init(elev=30, azim=60)
    if bounds:
        ax.set_xlim(bounds[0], bounds[3])
        ax.set_ylim(bounds[1], bounds[4])
        ax.set_zlim(bounds[2], bounds[5])



def plot_2d_projection(ax, data_test, kp_3d_pred, view_name, overlay_type='rgb', original_kps=None):
    """
    绘制2D投影对比图，包括原始关键点和预测关键点。
    
    :param ax: matplotlib的axes对象，用于绘图
    :param data_test: 测试数据字典，包含RGB、深度图、相机内外参等信息
    :param kp_3d_pred: 预测的3D关键点字典
    :param view_name: 视角名称（例如'front', 'left_shoulder'等）
    :param overlay_type: 可选'rgb'或'depth'，决定背景是RGB图像还是深度图
    :param original_kps: 原始2D关键点坐标列表，如果提供，则会画在图像上
    """
    def safe_transpose(arr):
        """安全转置函数（处理3D/4D输入）"""
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
            
        if arr.ndim == 4:  # (B,C,H,W)
            arr = arr[0]    # 取第一个样本
        elif arr.ndim == 3 and arr.shape[0] == 3:  # (C,H,W)
            pass
        else:
            arr = arr.squeeze()
        return arr.transpose(1,2,0)  # (H,W,C)
    rgb = safe_transpose(data_test[f'{view_name}_rgb'])  # (H,W,3)
    depth = data_test[f'{view_name}_depth']
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    depth = depth.squeeze()  # (H,W)
    intrinsics = data_test[f'{view_name}_camera_intrinsics'].squeeze()
    extrinsics = data_test[f'{view_name}_camera_extrinsics'].squeeze()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.cpu().numpy()
    H, W = depth.shape

    # 显示背景（添加分辨率适配）
    if overlay_type == 'rgb':
        img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1 else rgb.astype(np.uint8)
        if original_kps is not None:
            if isinstance(original_kps, torch.Tensor):
                original_kps = original_kps.cpu().numpy()
            original_kps = np.clip(original_kps, [0, 0], [W-1, H-1])
            ax.scatter(original_kps[:,0], original_kps[:,1], 
                      c='blue', s=20, marker='x', label='Original KPs')
        ax.imshow(img)
    else:
        ax.imshow(depth, cmap='gray', vmin=0, vmax=2)
    
    colors = plt.get_cmap('tab10').colors
    markers = ['*', 's', 'D', 'o']
    # 投影关键点
    for kp_id in range(1, 5):
        kp_world = kp_3d_pred[f'kp_{kp_id}']
        if isinstance(kp_world, torch.Tensor):
            kp_world = kp_world.cpu().numpy()
        # 处理可能的batch维度
        if kp_world.ndim == 3:  # (B,N,3)
            kp_world = kp_world[0]  # (N,3)
        # 世界→相机坐标系 (使用外参的逆变换)
        R = extrinsics[:3,:3]
        t = extrinsics[:3,3]
        # kp_cam = (kp_world - t) @ R.T
        # kp_homo = np.column_stack([kp_cam, np.ones(len(kp_cam))])
        # kp_pixel = kp_homo @ intrinsics.T
        kp_cam = (kp_world - t) @ R  # 等价于 R.T @ (kp_world - t)
        # 相机→像素坐标
        kp_pixel = kp_cam @ intrinsics.T
        kp_pixel = kp_pixel[:,:2] / kp_pixel[:,2:]
        # 过滤有效点
        valid_mask = (
            (kp_pixel[:,0] >= 0) & (kp_pixel[:,0] < W) &
            (kp_pixel[:,1] >= 0) & (kp_pixel[:,1] < H) &
            (kp_cam[:,2] > 0)  # z值必须为正（在相机前方）
        )
        kp_pixel = kp_pixel[valid_mask]
        # 绘制     当前视角的关键点，使用特殊标记
        current_view = (kp_id-1 == ['front', 'left_shoulder', 'right_shoulder', 'wrist'].index(view_name))
        label = f'KP_{kp_id}' + (' (current)' if current_view else '')
        
        if len(kp_pixel) > 0:
            ax.scatter(kp_pixel[:,1], kp_pixel[:,0], c=[colors[kp_id-1]], 
                       s=50 if current_view else 30, marker=markers[kp_id-1], alpha=0.8,
                      edgecolors='white', linewidth=0.5, label=label)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{view_name} Projection')



def plot_error_heatmap(ax, data_test, kp_3d_pred, view_name):
    """绘制投影误差热力图"""
    def safe_squeeze(arr):
        """安全压缩维度，保留至少2D"""
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr
    # 获取原始关键点和预测投影
    kp_idx = ['front', 'left_shoulder', 'right_shoulder', 'wrist'].index(view_name) + 1
    original_kps = safe_squeeze(data_test[f'kp_{kp_idx}'])
    scale = 128 / 518
    # 收集误差数据
    points = []
    errors = []
    for kp_id in range(1, 5):
        if kp_id-1 == ['front', 'left_shoulder', 'right_shoulder', 'wrist'].index(view_name):
            kp_world = safe_squeeze(kp_3d_pred[f'kp_{kp_id}'])
            extrinsics = safe_squeeze(data_test[f'{view_name}_camera_extrinsics'])
            intrinsics = safe_squeeze(data_test[f'{view_name}_camera_intrinsics'])
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            # kp_cam = (kp_world - t) @ R.T
            # kp_homo = np.column_stack([kp_cam, np.ones(len(kp_cam))])
            # kp_pixel = kp_homo @ intrinsics.T
            kp_cam = (kp_world - t) @ R
            kp_pixel = kp_cam @ intrinsics.T
            kp_pixel = kp_pixel[:,:2] / kp_pixel[:,2:]
            if len(kp_pixel) == len(original_kps):
                error = np.linalg.norm(kp_pixel - original_kps * scale, axis=1)
                points.append(kp_pixel)
                errors.append(error)
            else:
                print(f"Warning: Mismatched keypoints count ({len(kp_pixel)} vs {len(original_kps)})")

    # 绘制误差图
    if points and errors:
        points = np.concatenate(points, axis=0)  # (N,2)
        errors = np.concatenate(errors, axis=0)  # (N,)
        
        # 检查维度一致性
        assert len(points) == len(errors), \
               f"Points ({len(points)}) and errors ({len(errors)}) must have same length"
        # 方案1：带误差值的散点图
        sc = ax.scatter(points[:,0], points[:,1], c=errors,
                       cmap='Reds', s=30, alpha=0.7,
                       vmin=0, vmax=20)  # 设置合理的误差范围
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
        ax.set_title('Reprojection Errors')
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label('Error (pixels)')
        # 方案2：误差直方图（备用）
        # ax.hist(errors, bins=20, color='skyblue')
        # ax.set_xlabel('Error (pixels)')
        # ax.set_title('Error Distribution')
    else:
        ax.text(0.5, 0.5, 'No valid reprojections', 
               ha='center', va='center', transform=ax.transAxes)

        

def visualize_comparison(data_test, kp_3d_pred, save_dir="debug_runs/comparison_results"):
    """
    多模态比较预测关键点与原始点云
    包含：3D点云对比、2D投影对比、深度图叠加、误差分析
    """
    os.makedirs(save_dir, exist_ok=True)
    # ========== 0. 数据预处理（自动处理张量转换） ==========
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.array(data)
    
    # 转换所有输入数据
    data_test = {k: to_numpy(v) for k, v in data_test.items()}
    kp_3d_pred = {k: to_numpy(v) for k, v in kp_3d_pred.items()}

    # ================= 1. 3D点云对比可视化 =================
    fig = plt.figure(figsize=(18, 10))
    # 3D视图1
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_comparison(ax1, data_test, kp_3d_pred, view='overview')
    # 3D视图2（不同视角）
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_comparison(ax2, data_test, kp_3d_pred, view='top_down')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/3d_comparison.png", dpi=300)
    plt.close()
    # ================= 2. 各相机视角2D投影对比 =================
    view_mapping = {
        'front': 'front',
        'left_shoulder': 'left_shoulder',
        'right_shoulder': 'right_shoulder',
        'wrist': 'wrist'
    }
    original_keypoints = {
        'front': data_test['kp_1'].squeeze() * 128 / 518,
        'top': data_test['kp_2'].squeeze() * 128 / 518,
        'right': data_test['kp_3'].squeeze() * 128 / 518,
        # 'wrist': data_test['kp_4'].squeeze() * 128 / 518
    }    
    for view_name in view_mapping.keys():
        fig = plt.figure(figsize=(15, 6))
        # RGB图像
        ax1 = fig.add_subplot(131)
        plot_2d_projection(ax1, data_test, kp_3d_pred, view_name, overlay_type='rgb', original_kps=original_keypoints[view_name])
        ax1.set_title(f'{view_name} RGB Projection')
        # 深度图
        ax2 = fig.add_subplot(132)
        plot_2d_projection(ax2, data_test, kp_3d_pred, view_name, overlay_type='depth')
        ax2.set_title(f'{view_name} Depth Projection')
        # 误差热力图
        ax3 = fig.add_subplot(133)
        plot_error_heatmap(ax3, data_test, kp_3d_pred, view_name)
        ax3.set_title(f'{view_name} Error Heatmap')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/2d_{view_name}_comparison.png", dpi=300)
        plt.close()



def enhanced_visualization(sample_idx, view_idx, img_match, match_kpts_img, correspondence):
    # 新增调试输出
    print(f"\n=== Debug Sample {sample_idx}, View {view_idx} ===")
    
    # 数据提取（增加范围检查）
    img = img_match[sample_idx, view_idx].cpu().numpy().transpose(1,2,0)[..., 3:6]  # original (b, v, 10, 224, 224)
    kpts = match_kpts_img[sample_idx, :, view_idx].cpu().numpy()
    valid_mask = correspondence['valid_mask'][sample_idx, :, view_idx].cpu().numpy()
    
    # 新增：检查点坐标范围
    print(f"Valid points ratio: {valid_mask.mean():.2%}")

    # 创建画布（调整为3D可视化为主）
    fig = plt.figure(figsize=(24, 10))
    
    # === 子图1：3D点云与相机视角 ===
    ax1 = fig.add_subplot(131, projection='3d')
    points_3d = correspondence['original_3d'][sample_idx].cpu().numpy()
    camera_3d = correspondence['camera_3d'][sample_idx].cpu().numpy()
    
    # 绘制世界坐标系点云
    ax1.scatter(points_3d[valid_mask, 0], points_3d[valid_mask, 1], points_3d[valid_mask, 2],
               c='lime', s=20, alpha=0.8, label='Valid Points')
    ax1.scatter(points_3d[~valid_mask, 0], points_3d[~valid_mask, 1], points_3d[~valid_mask, 2],
               c='red', s=10, alpha=0.3, label='Invalid Points')
    
    # 绘制相机位置（修正实现）
    if camera_3d.ndim == 3:
        cam_pos = -np.mean(camera_3d[valid_mask, view_idx], axis=0)  # 相机在世界坐标系中的位置
        ax1.scatter(*cam_pos, c='blue', s=100, marker='^', label='Camera')
        ax1.quiver(*cam_pos, *np.mean(points_3d[valid_mask] - cam_pos, axis=0),
                 length=0.5, color='purple', arrow_length_ratio=0.1, label='View Direction')
    
    ax1.set_title('3D World Coordinates\n(Valid/Invalid Points & Camera Pose)')
    ax1.legend()

    # === 子图2：2D投影与图像叠加 ===
    ax2 = fig.add_subplot(132)
    ax2.imshow(img)
    ax2.scatter(kpts[valid_mask, 0], kpts[valid_mask, 1], 
               c='cyan', s=30, alpha=0.7, label='Projected Valid')
    ax2.scatter(kpts[~valid_mask, 0], kpts[~valid_mask, 1],
               c='magenta', s=15, alpha=0.3, label='Projected Invalid')
    ax2.set_title('2D Image Projection\n(Overlay on RGB Image)')
    ax2.legend()

    # === 子图3：相机坐标系点云 ===
    ax3 = fig.add_subplot(133, projection='3d')
    valid_cam = camera_3d[valid_mask, view_idx]
    invalid_cam = camera_3d[~valid_mask, view_idx]
    ax3.scatter(valid_cam[:, 0], valid_cam[:, 1], valid_cam[:, 2],
               c='yellow', s=20, alpha=0.8, label='Valid (Cam)')
    ax3.scatter(invalid_cam[:, 0], invalid_cam[:, 1], invalid_cam[:, 2],
               c='orange', s=10, alpha=0.3, label='Invalid (Cam)')
    ax3.set_title('3D Camera Coordinates\n(Transformed Points)')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(f"debug_runs/3view_sample{sample_idx}_view{view_idx}.png", dpi=300)
    plt.close()



def visualize_pc_transformation(pc_before, match_pc_before, pc_after, match_pc_after, 
                              title_before="Before", title_after="After", sample_idx=0):
    """
    可视化点云变换前后的对比
    参数:
        pc_before: 变换前的点云 (B,N,3) 或 (N,3)
        pc_after: 变换后的点云 (B,N,3) 或 (N,3)
        title_before: 前标题
        title_after: 后标题
        sample_idx: 要可视化的样本索引
    """
    # 提取指定样本的点云
    def extract_sample(pc):
        if isinstance(pc, list):
            return pc[sample_idx].cpu().numpy()
        return pc[sample_idx].cpu().numpy()
    
    pc_b = extract_sample(pc_before)
    match_b = extract_sample(match_pc_before)
    pc_a = extract_sample(pc_after)
    match_a = extract_sample(match_pc_after)
    
    # 创建画布
    fig = plt.figure(figsize=(20, 10))
    
    # 子图1：变换前
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pc_b[:,0], pc_b[:,1], pc_b[:,2], 
               c='blue', s=5, alpha=0.1, label='Task PC')
    ax1.scatter(match_b[:,0], match_b[:,1], match_b[:,2], 
               c='red', s=5, alpha=0.5, label='Match PC')
    ax1.set_title(f'{title_before}\nTask PC: {len(pc_b)}, Match PC: {len(match_b)}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 子图2：变换后
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pc_a[:,0], pc_a[:,1], pc_a[:,2], 
               c='blue', s=5, alpha=0.1, label='Task PC')
    ax2.scatter(match_a[:,0], match_a[:,1], match_a[:,2], 
               c='red', s=5, alpha=0.5, label='Match PC')
    ax2.set_title(f'{title_after}\nTask PC: {len(pc_a)}, Match PC: {len(match_a)}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 统一坐标轴范围和视角
    for ax in [ax1, ax2]:
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.view_init(elev=30, azim=45)  # 统一视角
    
    plt.tight_layout()
    plt.savefig(f"debug_runs/combined_{title_after.replace(' ','_')}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()


def gen_batch_ray_parellel(intrinsic,c2w,W,H):
    batch_size = intrinsic.shape[0]
    
    fx, fy, cx, cy = intrinsic[:,0,0].unsqueeze(1).unsqueeze(2), intrinsic[:,1,1].unsqueeze(1).unsqueeze(2), \
        intrinsic[:,0,2].unsqueeze(1).unsqueeze(2), intrinsic[:,1,2].unsqueeze(1).unsqueeze(2)
    i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  
    # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    i = i.unsqueeze(0).repeat(batch_size,1,1)
    j = j.unsqueeze(0).repeat(batch_size,1,1)
    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:,np.newaxis,np.newaxis, :3,:3], -1)  
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:, :3, -1].unsqueeze(1).unsqueeze(2).repeat(1,H,W,1)
    viewdir = rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
    return rays_d, rays_o, viewdir


# ===== StreamVGGT for processing dynamic view =====
def extract_streamvggt_features(rgb_vggt, model, device, return_attn=False):
     
    vggt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=vggt_dtype):
            images = rgb_vggt  # add batch dimension
            if return_attn:
                aggregated_tokens_list, ps_idx, attn = model.aggregator(images, return_attn=True)  # attn (B*S, num_heads, P, P) 全局注意力权重矩阵
            else:
                aggregated_tokens_list, ps_idx = model.aggregator(images, return_attn=False)
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_maps_by_unprojection = []
        for i in range(depth_map.size(0)):  
            point_map_by_unprojection = unproject_depth_map_to_point_map(
                depth_map[i].cpu().numpy(), # (V, 518, 518, 1)
                extrinsic[i].cpu().numpy(), # (V, 3, 4)
                intrinsic[i].cpu().numpy()
            )
            point_maps_by_unprojection.append(torch.from_numpy(point_map_by_unprojection).float())
        point_map_by_unprojection = torch.stack(point_maps_by_unprojection) # (B, V, 518, 518, 3)

        # point_map_view_1, point_map_view_2 = point_map[0, 0], point_map[0, 1]
        point_map_view_1 = point_map_by_unprojection[:,0,...].detach().clone().to(device)
        point_map_view_2 = point_map_by_unprojection[:,1,...].detach().clone().to(device)
        point_map_view_3 = point_map_by_unprojection[:,2,...].detach().clone().to(device)
        point_conf_view_1, point_conf_view_2, point_conf_view_3 = point_conf[:, 0], point_conf[:, 1], point_conf[:, 2]
        extrinsic_1, extrinsic_2, extrinsic_3 = extrinsic[:, 0], extrinsic[:, 1], extrinsic[:, 2]
        intrinsic_1, intrinsic_2, intrinsic_3 = intrinsic[:, 0], intrinsic[:, 1], intrinsic[:, 2]
        depth_pred_1, depth_pred_2 = depth_map[:, 0].squeeze(-1), depth_map[:, 1].squeeze(-1)
        depth_pred_3 = depth_map[:, 2].squeeze(-1)

        image_shape = tuple(rgb_vggt.shape[-2:])
        
        if return_attn:
            # 拆分为4个 (1, num_heads, P, P) 的自注意力权重 （14，23, 32, 41）
            cost_1, cost_2, cost_3, cost_4 = attn.chunk(4, dim=0)   
            cost_1 = cost_1.mean(dim=1)             # 多头注意力关注不同特征，取平均得到更鲁棒的相似度矩阵 (B, P, P)
            cost_2 = cost_2.mean(dim=1)
            cost_3 = cost_3.mean(dim=1)
            # cost_4 = cost_4.mean(dim=1)
            return {
                'point_map_view_1': point_map_view_1,   # (B, 518, 518, 3)
                'point_map_view_2': point_map_view_2,
                'point_map_view_3': point_map_view_3,
                # 'point_map_view_4': point_map_view_4,
                'point_conf_view_1': point_conf_view_1, # (B, 518, 518)
                'point_conf_view_2': point_conf_view_2,
                'point_conf_view_3': point_conf_view_3, 
                # 'point_conf_view_4': point_conf_view_4,
                'extrinsic_1': extrinsic_1,             # (B, 3, 4)
                'extrinsic_2': extrinsic_2,
                'extrinsic_3': extrinsic_3,
                # 'extrinsic_4': extrinsic_4,
                'intrinsic_1': intrinsic_1,             # (B, 3, 3)
                'intrinsic_2': intrinsic_2,
                'intrinsic_3': intrinsic_3,
                # 'intrinsic_4': intrinsic_4,
                'depth_pred_1': depth_pred_1,           # (B, 518, 518)
                'depth_pred_2': depth_pred_2,
                'depth_pred_3': depth_pred_3,
                # 'depth_pred_4': depth_pred_4,
                'image_shape': image_shape,
                'cost_1': cost_1,                       # (B, P, P)
                'cost_2': cost_2,
                'cost_3': cost_3,
                # 'cost_4': cost_4,
                # 'aggregated_tokens_list': aggregated_tokens_list,
                'images': images,                       # (B, V, 3, 518, 518)
                'ps_idx': ps_idx                        # 5
            }, aggregated_tokens_list

    return {
        'point_map_view_1': point_map_view_1,   # (B, 518, 518, 3)
        'point_map_view_2': point_map_view_2,
        'point_map_view_3': point_map_view_3,
        'point_conf_view_1': point_conf_view_1, # (B, 518, 518)
        # 'point_conf_view_2': point_conf_view_2,
        # 'point_conf_view_3': point_conf_view_3, 
        # 'point_conf_view_4': point_conf_view_4,
        'extrinsic_1': extrinsic_1,             # (B, 3, 4)
        'extrinsic_2': extrinsic_2,
        'extrinsic_3': extrinsic_3,
        # 'extrinsic_4': extrinsic_4,
        'intrinsic_1': intrinsic_1,             # (B, 3, 3)
        'intrinsic_2': intrinsic_2,
        'intrinsic_3': intrinsic_3,
        # 'intrinsic_4': intrinsic_4,
        'depth_pred_1': depth_pred_1,           # (B, 518, 518)
        'depth_pred_2': depth_pred_2,
        'depth_pred_3': depth_pred_3,
        # 'depth_pred_4': depth_pred_4
    }, image_shape, images, aggregated_tokens_list, ps_idx