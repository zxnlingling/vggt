import copy
import math
import time
import torch
import logging
from torch import nn
from functools import partial
from omegaconf import OmegaConf
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from hydra.utils import instantiate
if hasattr(torch.amp, 'autocast'):
    # PyTorch>=2.3
    from torch.amp import autocast
else:
    from torch.cuda.amp import autocast
from hydra import compose, initialize
from .avt_vggt_single import AVT_VGGT_Single
from hydra.core.global_hydra import GlobalHydra
from utils.mvt_utils import add_uni_noi, trans_pc
from utils.network_utils import LoRAQkv, LoRAConv2d
from utils.env_utils import CAMERAS, SCENE_BOUNDS, IMAGE_SIZE
from utils.vggt_utils import get_model_para, restore_cropped_image, save_rgb_images, get_depth, get_3d_preprocess, get_depth_st2


def build_vggt_custom_select(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    image_size=518,
    include_keys=None
):
    hydra_overrides = [         
        f"++configs.model.img_size={image_size}",
    ]

    # Read config and init model
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # Specify the config directory
    with initialize(config_path="..", version_base=None):
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.configs.model, _recursive_=True)
    assert isinstance(model, nn.Module), "Model is not a PyTorch module!"
    _load_checkpoint_select(model, ckpt_path, include_keys)
    model = model.to(device)

    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint_select(model, ckpt_path, include_keys=None):
    if ckpt_path is not None:
        # Load the checkpoint
        sd = torch.load(ckpt_path, map_location="cpu")

        # If specific keys are to be included, filter the state dictionary
        if include_keys is not None:
            sd = {k: v for k, v in sd.items() if any(k.startswith(key) for key in include_keys)}

        if "aggregator.patch_embed.pos_embed" in sd:
            del sd["aggregator.patch_embed.pos_embed"]

        # Load the filtered state dictionary into the model
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

        if missing_keys:
            logging.debug(f"Missing keys: {missing_keys}")  # Suppressed to debug level
        if unexpected_keys:
            logging.debug(f"Unexpected keys in checkpoint: {unexpected_keys}")  # Suppressed to debug level

        logging.info("Loaded checkpoint successfully")


class LoRA_VGGT(nn.Module):
    """Applies low-rank adaptation to a VGGT model's aggregator and depth head.

    Args:
        vggt_model: a VGGT model
        r: rank of LoRA
        alpha: scaling factor for LoRA adaptation
        rank_for: "channels" or "kernel", controls LoRA decomposition strategy

    """
    def __init__(self, vggt_model: VGGT, r: int, alpha: float = 1.0, rank_for: str = 'channels'):

        super(LoRA_VGGT, self).__init__()
        assert isinstance(vggt_model, VGGT), "vggt_model must be a VGGT instance"
        assert r > 0, "LoRA rank must be positive"

        self.model = vggt_model

        self.model.aggregator.patch_embed.pos_embed = nn.Parameter(
            torch.randn(1, 257, 1024)
        )

        # Define trainable prefixes
        self.trainable_prefixes = [
            'aggregator.camera_token',
            'aggregator.register_token',
            'aggregator.patch_embed.cls_token',
            'aggregator.patch_embed.pos_embed',
            
            # patch_embed layers 
            *[f'aggregator.patch_embed.blocks.{i}.mlp' for i in range(20, 24)],
            'aggregator.patch_embed.patch_embed.proj',  
            
            # frame_blocks and global_blocks attention and MLP
            *[f'aggregator.frame_blocks.{i}.mlp' for i in range(20, 24)],
            *[f'aggregator.global_blocks.{i}.mlp' for i in range(20, 24)],
        ]

        # Add LoRA adapters to attention blocks (query, value)
        combined_blocks = list(self.model.aggregator.patch_embed.blocks) + list(self.model.aggregator.frame_blocks) \
            + list(self.model.aggregator.global_blocks)
        assign_lora = partial(LoRAQkv, rank=r, alpha=alpha)
        self._apply_lora_to_blocks(combined_blocks, assign_lora)
        # self._apply_lora_to_depth_head(module=self.model.depth_head, rank=r, alpha=alpha, rank_for=rank_for)

        # After applying LoRA, freeze all params then unfreeze specific ones
        self._setup_trainable_parameters(combined_blocks)

        self.vggt = self.model

    def _apply_lora_to_blocks(self, blocks: nn.ModuleList, assign_lora):
        
        for block in blocks:
            block.attn.qkv = assign_lora(block.attn.qkv)

    def _setup_trainable_parameters(self, combined_blocks):
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA layers (q and v projections)
        for block in combined_blocks:
            for param in block.attn.qkv.lora_q.parameters():
                param.requires_grad = True
            for param in block.attn.qkv.lora_v.parameters():
                param.requires_grad = True

        # Unfreeze depth head LoRA parameters
        # for name, param in self.model.depth_head.named_parameters():
        #     if 'delta_weight_A' in name or 'delta_weight_B' in name:
        #         param.requires_grad = True

        # Unfreeze parameters matching trainable_prefixes
        for name, param in self.model.named_parameters():
            if any(name.startswith(prefix) for prefix in self.trainable_prefixes):
                param.requires_grad = True
            if ('norm' in name or 'ls.' in name) and not any(name.startswith(prefix) for prefix in self.trainable_prefixes):
                param.requires_grad = False

    def _apply_lora_to_depth_head(self, module, rank, alpha, rank_for):

        for param in module.parameters():
            param.requires_grad = False

        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                lora_config = {
                    'alpha': alpha,
                    'rank': rank,
                    'rank_for': rank_for  
                }
                setattr(module, name, LoRAConv2d(child, lora_config=lora_config))
            else:
                self._apply_lora_to_depth_head(child, rank=rank, alpha=alpha, rank_for=rank_for)
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vggt, name)


class AVT_VGGT(nn.Module):
    def __init__(
        self,
        depth,
        num_rot,
        attn_dim,
        feat_dim,
        lang_dim,
        lang_len,
        img_aug_2,
        activation,
        attn_heads,
        im_channels,
        proprio_dim,
        wpt_img_aug,
        attn_dropout,
        img_feat_dim,
        attn_dim_head,
        lora_finetune,
        st_wpt_loc_aug,
        weight_tie_layers,
        st_wpt_loc_inp_no_noise,
        lora_r,
        st_sca,
        add_corr,
        img_size,
        add_depth,
        norm_corr,
        stage_two,
        use_renderer,
        add_pixel_loc,
        img_patch_size,
        flash_attention,
        render_view_num,
        use_ray_renderer,
        image_resolution,
        vggt_config,
        vggt_ckpt,
        rank,
        renderer_device="cuda:0",
    ):
        """Arbitrary-View Transfomer based on Visual Geometry Grounded Transformer 
        :param st_sca: scaling of the 3D scene in the second stage
        :param stage_two: whether or not there are two stages
        :param use_renderer: whether to use renderer in the first stage
        :param flash_attention: use flash attention for fast training (control in RVT-2 action predictor)
        :param render_view_num: number of rendered views from VGGT
        :param use_ray_renderer: whether to use ray renderer or rvt renderer
        :param image_resolution: image size in VGGT
        
        """
        super().__init__()

        self.device = renderer_device
        from renderers.rvt_renderer import RVTBoxRenderer as Renderer
        global Renderer

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]
        del args["stage_two"]
        del args["img_aug_2"]

        self.num_rot = num_rot
        self.stage_two = stage_two
        self.st_sca = st_sca
        self.st_wpt_loc_aug = st_wpt_loc_aug
        self.st_wpt_loc_inp_no_noise = st_wpt_loc_inp_no_noise
        self.img_aug_2 = img_aug_2
        self.use_renderer = use_renderer
        self.vggt_ckpt = vggt_ckpt
        self.image_resolution = image_resolution
        self.img_patch_size = img_patch_size
        # for verifying the input
        self.img_feat_dim = img_feat_dim
        self.proprio_dim = proprio_dim
        lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = Renderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            three_views=True,
            with_depth=add_depth,
        )
        self.render_num_img = 3
        
        self.num_img = len(CAMERAS) # for the first stage
        self.img_size = img_size

        vggt = build_vggt_custom_select(vggt_config, vggt_ckpt, device=f"cuda:{rank}",
                                        include_keys=['aggregator'], #, 'depth_head'
                                        image_size=image_resolution)  
        if lora_finetune:    
            lora_vggt = LoRA_VGGT(vggt, lora_r)
            self.vggt = lora_vggt.vggt
            if rank == 0:
                get_model_para(self.vggt)
                
        else:
            self.vggt = vggt.eval()
            for param in self.vggt.parameters():
                param.requires_grad = False            
            if rank == 0:
                get_model_para(self.vggt)

        self.vggt1 = AVT_VGGT_Single(
            **args,
            renderer=self.renderer,
            no_feat=self.stage_two,
            vggt=self.vggt
        )
        if self.stage_two:
            self.vggt2 = AVT_VGGT_Single(**args, renderer=self.renderer, vggt=self.vggt)
    

    def get_pt_loc_on_img(self, pt, vggt1_or_vggt2, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local  
        :param out: output from avt_vggt 

        """
        assert len(pt.shape) == 3
        bs, _np, x = pt.shape
        assert x == 3

        assert isinstance(vggt1_or_vggt2, bool)
        if vggt1_or_vggt2:
            assert out is None
            out = self.vggt1.renderer.get_pt_loc_on_img(
                pt, fix_cam=True, dyn_cam_info=dyn_cam_info
            )
        else:
            assert self.stage_two
            assert out is not None
            assert out['wpt_local1'].shape == (bs, 3)
            pt, _ = trans_pc(pt, loc=out["wpt_local1"], sca=self.st_sca)
            pt = pt.view(bs, _np, 3)
            out = self.vggt2.renderer.get_pt_loc_on_img(
                    pt, fix_cam=True, dyn_cam_info=dyn_cam_info
                )
        return out # wpt_img



    def get_wpt(self, out, vggt1_or_vggt2, intrinsics=None, extrinsics=None):
        """
        Estimate the q-values given output from avt_vggt
        :param out: output from avt_vggt

        """
        assert isinstance(vggt1_or_vggt2, bool)
        # print("[DEBUG] vggt1_or_vggt2 = ", vggt1_or_vggt2)
        if vggt1_or_vggt2:
            wpt = self.vggt1.get_wpt(
                out, vggt1_or_vggt2, intrinsics, extrinsics, 
            )                
        else:
            assert self.stage_two
            wpt = self.vggt2.get_wpt(
                out["vggt2"], 
            )
            wpt = out["rev_trans"](wpt)

        return wpt

    
    def render(self, pc, img_feat, img_aug, vggt1_or_vggt2, pixel_positions=None):
        """
        Args:
            vggt1_or_vggt2: True for the first stage while False for the second
            pixel_positions: used only when self.use_renderer == False 
        Returns:
            img: (b, v, C, H, W)
        """
        assert isinstance(vggt1_or_vggt2, bool)

        if vggt1_or_vggt2:
            vggt = self.vggt1
        else:
            vggt = self.vggt2

        if vggt1_or_vggt2 and not self.use_renderer:
            # if not use renderer (in stage 1)
            assert pixel_positions is not None, "pixel_positions must be provided when use_renderer=False"
            img, pc_img = restore_cropped_image(img_feat, pixel_positions, points=pc)   # (b, 4, 3, 128, 128)

            # ground truth depth
            depth_maps, valid_mask = get_depth(pixel_positions, pc)

            combined_feat = torch.cat([pc_img, img, depth_maps], dim=2)                 # (b, v, 3+3, H, W)
            # Raw img: 0 ~ 0.9843137860298157
            # Raw pc: -0.9999954700469971 ~ 0.9999979734420776

            # apply multimodal noise
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=combined_feat.device)
                noise = stdv * ((2 * torch.rand(*img.shape, device=combined_feat.device)) - 1)
                combined_feat[:, :, :3, :, :] = torch.clamp(combined_feat[:, :, :3, :, :] + noise, -1, 1)
                stdv_pc = img_aug * torch.rand(1, device=combined_feat.device)
                noise_pc = stdv_pc * ((2 * torch.rand(*pc_img.shape, device=combined_feat.device)) - 1)
                combined_feat[:, :, 3:, :, :] = torch.clamp(combined_feat[:, :, 3:, :, :] + noise_pc, -1, 1)

            img = combined_feat

        else:
            # use renderer
            with torch.no_grad():
                with autocast(device_type="cuda", enabled=False):
                    if vggt.add_corr:
                        if vggt.norm_corr:
                            img = []
                            for _pc, _img_feat in zip(pc, img_feat):
                                # fix when the pc is empty
                                max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
                                img.append(
                                    self.renderer(
                                        _pc,
                                        torch.cat((_pc / max_pc, _img_feat), dim=-1),
                                        fix_cam=True,
                                        dyn_cam_info=None,
                                    ).unsqueeze(0)
                                )
                        else:
                            img = [
                                self.renderer(
                                    _pc,
                                    torch.cat((_pc, _img_feat), dim=-1),
                                    fix_cam=True,
                                    dyn_cam_info=None,
                                ).unsqueeze(0)
                                for (_pc, _img_feat) in zip(pc, img_feat)
                            ]
                    else:
                        img = [
                            self.renderer(
                                _pc,
                                _img_feat,
                                fix_cam=True,
                                dyn_cam_info=None,
                            ).unsqueeze(0)
                            for (_pc, _img_feat) in zip(pc, img_feat)
                        ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)                        # (b, render_view_num, 7, 224, 224)
            depth_maps, valid_mask = get_depth_st2(img)

            # for visualization purposes
            if vggt.add_corr:
                vggt.img = img[:, :, 3:].clone().detach()           # (b, render_view_num, 4, 224, 224)
            else:
                vggt.img = img.clone().detach()

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

        if vggt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = vggt.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )
            # Raw pixel_loc: -1.0 ~ 1.0

        return img, depth_maps, valid_mask
    

    def verify_inp(
        self,
        pc,
        img_feat,
        pixel_positions,
        proprio,
        lang_emb,
        img_aug,
        wpt_local,
        rot_x_y,
        intrinsics,
        extrinsics,
    ):
        
        bs = len(pc)
        assert bs == len(img_feat) == len(pixel_positions) 
        assert len(intrinsics) == len(extrinsics) == self.num_img

        if not self.training:
            # no img_aug when not training
            assert img_aug == 0
            assert rot_x_y is None, f"rot_x_y={rot_x_y}"

        if self.training:
            assert (
                not wpt_local is None
            )

            assert rot_x_y.shape == (bs, 2), f"rot_x_y.shape={rot_x_y.shape}"
            assert (rot_x_y >= 0).all() and (
                rot_x_y < self.num_rot
            ).all(), f"rot_x_y={rot_x_y}"

        for _pc, _img_feat, _pixel_position in zip(pc, img_feat, pixel_positions):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape
            np3, xp = _pixel_position.shape

            assert np == np2 == np3
            assert x1 == 3
            assert x2 == xp == self.img_feat_dim

        for _intr, _extr in zip(intrinsics, extrinsics):
            dim_i1, dim_i2, dim_i3 = _intr.shape
            dim_e1, dim_e2, dim_e3 = _extr.shape

            assert dim_i1 == dim_e1 == bs
            assert dim_i2 == dim_i3 == 3
            assert dim_e2 == dim_e3 == 4

        bs3, x3 = proprio.shape
        assert bs == bs3
        assert (
            x3 == self.proprio_dim
        ), f"Does not support proprio of shape {proprio.shape}"

        bs4, x4, x5 = lang_emb.shape
        assert bs == bs4
        assert (
            x4 == self.lang_max_seq_len
        ), f"Does not support lang_emb of shape {lang_emb.shape}"
        assert (
            x5 == self.lang_emb_dim
        ), f"Does not support lang_emb of shape {lang_emb.shape}"

        if not (wpt_local is None):
            bs5, x6 = wpt_local.shape
            assert bs == bs5
            assert x6 == 3, f"Does not support wpt_local of shape {wpt_local.shape}"

        if self.training:
            assert (not self.stage_two) or (not wpt_local is None)

    
    def forward(
        self,
        pc,
        img_feat,
        pixel_positions,
        match_input_dict,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        wpt_local=None,
        rot_x_y=None,
        rank=0,
        intrinsics=None,
        extrinsics=None,
        iteration=0,
        epoch=0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape (num_points, img_feat_dim)
        :param pixel_positions: list tensors, each tensor of shape (num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        :param wpt_local: gt location of the wpt in 3D, tensor of shape (bs, 3)
        :param rot_x_y: (bs, 2) rotation in x and y direction
        :param intrinsics: list of tensors, each tensor of shape (bs, 3, 3)
        :param extrinsics: list of tensors, each tensor of shape (bs, 4, 4)
        :param rev_trans: set of functions, transforming wpt_local to global

        """

        t_start_vggt0 = time.time()
        self.verify_inp(        
            pc=pc,
            img_feat=img_feat,
            pixel_positions=pixel_positions,
            proprio=proprio,
            lang_emb=lang_emb,
            img_aug=img_aug,
            wpt_local=wpt_local,
            rot_x_y=rot_x_y,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        with torch.no_grad():
            if self.training and (self.img_aug_2 != 0):
                for x in img_feat:
                    stdv = self.img_aug_2 * torch.rand(1, device=x.device)
                    # values in [-stdv, stdv]
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise # TODO: the noise is not added to img_feat
            
            # (b, 3, 10, 224, 224) or (b, 4, 9, 128, 128)        
            img, depth_maps, valid_mask = self.render(pc=pc, img_feat=img_feat, img_aug=img_aug, vggt1_or_vggt2=True, pixel_positions=pixel_positions)

        if self.training:
            wpt_local_stage_one = wpt_local
            wpt_local_stage_one = wpt_local_stage_one.clone().detach()
        else:
            wpt_local_stage_one = wpt_local

        # t_end_vggt0 = time.time()
        # if iteration == 0:
        #     import os
        #     from utils.vggt_utils import visualize_point_cloud
        #     save_dir_3d = "debug_runs/images/temp_stage2/True/point_clouds"
        #     os.makedirs(save_dir_3d, exist_ok=True)
        #     for i in range(min(5, len(pc))):  # 保存前5个样本
        #         sample_pc = pc[i].cpu().numpy()
        #         save_path = os.path.join(save_dir_3d, f"st1_sample{i}.png")
        #         visualize_point_cloud(sample_pc, title=f"Original PC Sample {i}", save_path=save_path)
        # if rank == 0:
        #     if self.use_renderer:
        #         print("[DEBUG] AVT_VGGT rendering finished. Time Cost: {} minutes".format((t_end_vggt0 - t_start_vggt0) / 60.0))
        #     else:
        #         print("[DEBUG] AVT_VGGT recovering rgb views finished. Time Cost: {} minutes".format((t_end_vggt0 - t_start_vggt0) / 60.0))
        #     print("[DEBUG] Start timing for AVT_VGGT_Single forward ...")
        
        out = self.vggt1(
            img=img,
            proprio=proprio,
            lang_emb=lang_emb,
            wpt_local=wpt_local_stage_one,
            rot_x_y=rot_x_y,
            rank=rank,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            iteration=iteration,
            epoch=epoch,
            match_input_dict=match_input_dict,
            **kwargs,
        )

        # gt_depth_resized, valid_mask_resized = resize_gt_depth(depth_maps, valid_mask, size=self.image_resolution)
        # out["gt_depth"] = gt_depth_resized   
        # out["valid_mask"] = valid_mask_resized
        # t_end_1 = time.time()
        # if rank == 0:
        #     save_rgb_images(img[:, :, :3, :, :], "debug_runs/images/stage1", prefix="original", views=CAMERAS)
            # print("[DEBUG] vggt1 forward finished. Time Cost: {} minutes".format((t_end_1 - t_end_vggt0) / 60.0))

        if self.stage_two:
            # t_start_2 = time.time()
            with torch.no_grad():
                # adding then noisy location for training
                if self.training:
                    # noise is added so that the wpt_local2 is not exactly at the center of the pc
                    wpt_local_stage_one_noisy = add_uni_noi(        # (b, 3)
                        wpt_local_stage_one.clone().detach(), 2 * self.st_wpt_loc_aug
                    )
                    pc, rev_trans_st2 = trans_pc(
                        pc, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                    )   

                    if self.st_wpt_loc_inp_no_noise:
                        wpt_local2, _ = trans_pc(                   # (b, 3)
                            wpt_local, loc=wpt_local_stage_one_noisy, sca=self.st_sca
                        )
                    else:
                        wpt_local2, _ = trans_pc(
                            wpt_local, loc=wpt_local_stage_one, sca=self.st_sca
                        )
                else:
                    # bs, 3
                    wpt_local = self.get_wpt(
                        out, vggt1_or_vggt2=True, intrinsics=intrinsics, extrinsics=extrinsics
                    )
                    pc, rev_trans_st2 = trans_pc(
                        pc, loc=wpt_local, sca=self.st_sca
                    )
                    # bad name!
                    wpt_local_stage_one_noisy = wpt_local

                    # must pass None to vggt2 while in eval
                    wpt_local2 = None

                img, depth_maps_st, valid_mask_st = self.render(
                    pc=pc,
                    img_feat=img_feat,
                    img_aug=img_aug,
                    vggt1_or_vggt2=False,
                )
                # if iteration == 0:
                #     save_dir_3d = "debug_runs/images/temp_stage2/False/point_clouds"
                #     os.makedirs(save_dir_3d, exist_ok=True)
                #     for i in range(min(5, len(pc))):  # 保存前5个样本
                #         sample_pc = pc[i].cpu().numpy()
                #         save_path = os.path.join(save_dir_3d, f"st2_sample{i}.png")
                #         visualize_point_cloud(sample_pc, title=f"Original PC Sample {i}", save_path=save_path)
                    
                # if rank == 0:
                #     save_rgb_images(img, "debug_runs/images/stage2_final", prefix="final", views=CAMERAS)                          
            # t_end_2 = time.time()
            # print("[DEBUG] vggt2 rendering finished. Time Cost: {} minutes".format((t_end_2 - t_start_2) / 60.0))

            # t_start_3 = time.time()
            out_vggt2 = self.vggt2(
                img=img,
                proprio=proprio,
                lang_emb=lang_emb,
                wpt_local=wpt_local2,
                rot_x_y=rot_x_y,
                rank=rank,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                iteration=iteration,
                epoch=epoch,
                **kwargs,
            )
            # t_end_3 = time.time()
            # if rank == 0:
            #     print("[DEBUG] stage 2 vggt1 forward finished. Time Cost: {} minutes".format((t_end_3 - t_start_3) / 60.0))
            # gt_depth_resized_st, valid_mask_resized_st = resize_gt_depth(depth_maps_st, valid_mask_st, size=self.image_resolution)
            # out_vggt2["gt_depth2"] = gt_depth_resized_st   
            # out_vggt2["valid_mask2"] = valid_mask_resized_st

            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans_st2
            out["vggt2"] = out_vggt2
        
        
        return out
    
    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()