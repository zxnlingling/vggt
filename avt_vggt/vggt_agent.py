import clip
import time
import torch
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb
if hasattr(torch.amp, 'autocast'):
    # PyTorch>=2.3
    from torch.amp import autocast
else:
    from torch.cuda.amp import autocast
if hasattr(torch.amp, 'GradScaler'):
    # PyTorch 2.3+ 
    from torch.amp import GradScaler
else:
    from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR  
from utils.log_utils import manage_loss_log, manage_eval_log
from utils.trainer_utils import Lamb, GradualWarmupScheduler   
from utils.env_utils import CAMERAS, SCENE_BOUNDS, ActResult              
from torch.nn.parallel.distributed import DistributedDataParallel
from utils.vggt_utils import (get_image_augmentation, get_pt_loc_on_img, 
                              get_pc_img_feat_with_positions, move_pc_in_bound_with_positions, sigmoid)
from utils.mvt_utils import (stack_on_channel, place_pc_in_cube, generate_hm_from_pt, apply_se3_aug_con, 
                             normalize_quaternion, quaternion_to_discrete_euler, discrete_euler_to_quaternion, _clip_encode_text)


class VGGTAgent:
    def __init__(
        self,
        network: nn.Module, 
        lr: float,
        img_aug: bool,
        stage_two: bool,
        warmup_steps: int,
        use_input_pc: bool,
        optimizer_type: str,
        cos_dec_max_step: int, 
        image_resolution: int, 
        lambda_weight_l2: float,
        num_rotation_classes: int, 
        transform_augmentation: bool, 
        transform_augmentation_xyz: list, 
        transform_augmentation_rpy: list, 
        log_dir="",
        amp: bool = True,
        bnb: bool = True,
        rot_x_y_aug: int = 2,
        lr_cos_dec: bool = True,
        cameras: list = CAMERAS,
        gt_hm_sigma: float = 1.5, 
        move_pc_in_bound: bool = True, 
        scene_bounds: list = SCENE_BOUNDS, 
    ):
        
        self._lr = lr
        self.amp = amp 
        self.bnb = bnb
        self.cameras = cameras
        self.img_aug = img_aug
        self.log_dir = log_dir
        self._network = network 
        self.stage_two = stage_two
        self.lr_cos_dec = lr_cos_dec
        self.gt_hm_sigma = gt_hm_sigma
        self.rot_x_y_aug = rot_x_y_aug
        self.use_input_pc = use_input_pc
        self.warmup_steps = warmup_steps
        self.scene_bounds = scene_bounds
        self._optimizer_type = optimizer_type 
        self.cos_dec_max_step = cos_dec_max_step
        self.move_pc_in_bound = move_pc_in_bound
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        if hasattr(torch.amp, 'GradScaler'):
            self.scaler = GradScaler("cuda", enabled=self.amp)
        else:
            self.scaler = GradScaler(enabled=self.amp)
        self._num_rotation_classes = num_rotation_classes
        self.num_all_rot = self._num_rotation_classes * 3
        self._transform_augmentation = transform_augmentation
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz)) 
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        # TODO: Set different lr for different sets of parameters

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    [p for p in self._network.parameters() if p.requires_grad],
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    [p for p in self._network.parameters() if p.requires_grad],
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                [p for p in self._network.parameters() if p.requires_grad],
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        elif self._optimizer_type == "adamw":
            print("Using AdamW optimizer of VGGT")
            self._optimizer = torch.optim.AdamW(
                [p for p in self._network.parameters() if p.requires_grad],
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            print("Unsupported optimizer.")
    
        
        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None

        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )


    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """
        copied from SAM2Act

        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )
    
    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of vggt1 or vggt2
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for training and preduction

        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, concatenate the q_trans, and replace all other q
            if self.stage_two:
                out = out["vggt2"]
                q_trans2 = out["trans"].view(bs, 3, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            q_trans = None
            if self.stage_two:
                out = out["vggt2"]

        rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]), dim=-1).view(bs, -1)
        grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
        collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]

        return q_trans, rot_q, grip_q, collision_q
    
    # TODO: check this function
    def preprocess_images(self, replay_sample, cameras): 
        """
        Input params: 
            replay_sample["%s_rgb" % n]: input rgb images
                                        each view of shape (b, 1, 3, 128, 128)
            cameras: views
            use_input_pc: (bool) whether use the original point cloud
                        use_input_pc = False means using point map predicted by the pretrained VGGT model
            device: where to load the pretrained VGGT model
        Output params:
            obs: list with length 4, obs[0][0] is rgb of shape (b, 3, 128, 128) or (b, 3, 182, 182)
            pcds: list with length 4, pcd[0] is pcd of shape (b, 3, 128, 128) or (b, 3, 182, 182)

        """
        obs, pcds = [], []
        intrinsics, extrinsics = [], []
        original_rgbs = {}
        if self.use_input_pc:
            for n in cameras:
                rgb = stack_on_channel(replay_sample["%s_rgb" % n])             # (B, 3, 128, 128) 
                original_rgbs[n] = rgb.clone().detach()
                # point cloud
                pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])     # (B, 3, 128, 128)
                rgb = (rgb.float() / 255.0) * 2.0 - 1.0                                      
                obs.append([rgb, pcd])
                pcds.append(pcd) 
                intrinsic = stack_on_channel(replay_sample["%s_camera_intrinsics" % n]) # (B, 3, 3)
                extrinsic = stack_on_channel(replay_sample["%s_camera_extrinsics" % n]) # (B, 4, 4)
                intrinsics.append(intrinsic)
                extrinsics.append(extrinsic)
        else:
            model = self._network.vggt
            rgb_imgs = []
            for n in cameras:
                rgb = replay_sample["%s_rgb" % n]                               # torch.Size([B, 1, 3, 128, 128]) 
                rgb_imgs.append(rgb)
            rgb_tensor = torch.cat(rgb_imgs, dim=1)                             # torch.Size([B, 4, 3, 128, 128])
            rgb_per_sample = []
            for i in range(rgb_tensor.shape[0]):
                img = rgb_tensor[i]                                             # torch.Size([4, 3, 128, 128])
                # resize for VGGT (518 x 518) 
                resized_img = torch.nn.functional.interpolate(                  # torch.Size([4, 3, 182, 182])
                    img,
                    size=(self._image_resolution, self._image_resolution),                
                    mode='bicubic',
                    align_corners=False
                )                                       
                resized_img = resized_img.float() / 255.0
                rgb_per_sample.append(resized_img)
            rgb_samples = torch.stack(rgb_per_sample)                           # torch.Size([B, 4, 3, 182, 182])
            # REQUIRED by VGGT forward(): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
            aggregated_tokens_list, ps_idx = model.aggregator(rgb_samples)
            # Predict Point Maps    point_map: (B, 4, 182, 182, 3), point_conf: (B, 4, 182, 182)
            point_map, point_conf = model.point_head(aggregated_tokens_list, rgb_samples, ps_idx)
            for i in range(len(cameras)):
                print("[DEBUG] point_map[:,i].shape = ", point_map[:,i].shape)
                pcd = point_map[:,i].permute(0, 3, 1, 2)
                obs.append([rgb_samples[:,i], pcd])
                pcds.append(pcd)
        return obs, pcds, intrinsics, extrinsics, original_rgbs
    

    
    
    def calculate_matching_loss(self, out, point_map_view_1, point_map_view_2, point_map_view_3, point_map_view_4, 
                                kp_1, kp_2, kp_3, kp_4, thres3d_neg = 0.1):

        desc_1, desc_2, desc_3, desc_4 = out["feature_1"], out["feature_2"], out["feature_3"], out["feature_4"]  # (B, N, C)
        
        def prepare_point_map(p_map, kp):
            batch_size = len(kp)
            max_kp = max(k.shape[0] for k in kp)

            pts3d = torch.zeros(batch_size, max_kp, 3, device=p_map.device)
            kp_mask = torch.zeros(batch_size, max_kp, dtype=torch.bool, device=p_map.device)

            for i, k in enumerate(kp):
                kp_mask[i, :k.shape[0]] = True
                h, w = p_map.shape[1:3]
                y_coords = torch.clamp(kp[i][:, 1].long(), 0, h-1)
                x_coords = torch.clamp(kp[i][:, 0].long(), 0, w-1)
                pts3d[i, :k.shape[0]] = p_map[i, y_coords, x_coords]

            return pts3d  # (B, N, 3)
        
        # 获取3D点坐标（518分辨率）
        pts3d_1 = prepare_point_map(point_map_view_1, kp_1)
        pts3d_2 = prepare_point_map(point_map_view_2, kp_2)
        pts3d_3 = prepare_point_map(point_map_view_3, kp_3)
        pts3d_4 = prepare_point_map(point_map_view_4, kp_4)
        
        def compute_ap(desc_a, desc_b, pts3d_a, pts3d_b):
            B, N, _ = desc_a.shape
            eye_mask = torch.eye(N, device=self._device).bool().unsqueeze(0)
            neg_mask = (torch.cdist(pts3d_a, pts3d_b) > thres3d_neg) & ~eye_mask
            sim = torch.bmm(desc_a, desc_b.transpose(-1, -2))  # (B, N, N)
            
            pos_idxs = torch.stack([
                torch.zeros(N, dtype=torch.long, device=self._device),
                torch.arange(N, device=self._device),
                torch.arange(N, device=self._device)
            ], dim=1)
            
            pos_sim = sim[pos_idxs[:,0], pos_idxs[:,1], pos_idxs[:,2]]
            rpos = torch.sigmoid((1. - pos_sim)/0.01) + 1
            rall = rpos + torch.sum(
                torch.sigmoid((sim[pos_idxs[:,0], pos_idxs[:,1]] - 1.)/0.01) * 
                neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),
                dim=-1
            )
            return rpos / rall

        ap1 = compute_ap(desc_1, desc_2, pts3d_1, pts3d_2)
        ap2 = compute_ap(desc_2, desc_3, pts3d_2, pts3d_3)
        ap3 = compute_ap(desc_3, desc_4, pts3d_3, pts3d_4)
        ap4 = compute_ap(desc_4, desc_1, pts3d_4, pts3d_1)
        
        ap = (ap1 + ap2 + ap3 + ap4) / 4
        ap_loss = torch.mean(1. - ap)
        
        return ap_loss


    def update(
        self,
        step: int,
        rank: int,
        epoch: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        
        # if rank == 0:
        #     print("[DEBUG] Start updating our agent ...")
        # t_start_1 = time.time()
        
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (1, self._net_mod.proprio_dim,)

        action_rot_grip = replay_sample["rot_grip_action_indicies"][:, -1].int()    # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][:, -1].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]                  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]                              # (b, 3)
        action_rot = action_gripper_pose[:, 3:7]                                    # (b, 4)
        action_grip = action_rot_grip[:, -1]                                        # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = stack_on_channel(replay_sample["low_dim_state"])                  # (b, 18)
        return_out = {}

        obs, pcd, train_intrinsics, train_extrinsics, original_rgbs = self.preprocess_images(replay_sample, self.cameras)    
        vggt_features = replay_sample['vggt_features']
        kp_1, kp_2, kp_3, kp_4 = replay_sample['kp_1'], replay_sample['kp_2'], replay_sample['kp_3'], replay_sample['kp_4']
        resized_images = replay_sample["resized_images"].transpose(1, 0)
        match_input_dict = {
            'rgbs': resized_images.transpose(0, 1), # (B, 4, 3, 518, 518) CHW
            'kp_1': kp_1, # (B, 300, 2)
            'kp_2': kp_2,
            'kp_3': kp_3,
            'kp_4': kp_4,
            'normalize': True,
        }

        # t_end_1 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Processed replay samples. Time Cost: {} minutes".format((t_end_1 - t_start_1) / 60.0))
         
        with torch.no_grad():
            # get pc and rgb features of shape (len b, H * W * 4, 3)
            pc, img_feat, pixel_positions, colors = get_pc_img_feat_with_positions(obs, pcd, original_rgbs)

            if self._transform_augmentation and backprop:
                # apply SE3 augmentation to a point clouds and actions
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range = self._transform_augmentation_xyz.detach().clone(),# torch.tensor(self._transform_augmentation_xyz)
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con, device = pc.device)
                action_rot = torch.tensor(action_rot, device = pc.device)

            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat, pixel_positions, colors = move_pc_in_bound_with_positions(
                pc, img_feat, self.scene_bounds, colors, pixel_positions, no_op=not self.move_pc_in_bound
            )

            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = place_pc_in_cube(_pc, _wpt, with_mean_or_bounds=False, scene_bounds = self.scene_bounds,)
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            pc = [place_pc_in_cube(_pc, with_mean_or_bounds = False, scene_bounds = self.scene_bounds,)[0] for _pc in pc]
            
            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                # Initialize VGGT default image augmentations (color jitter, grayscale, gaussian blur)
                img_aug = get_image_augmentation(color_jitter = None, gray_scale = True, gau_blur = False,)
            else:
                img_aug = 0


        with autocast(device_type="cuda", enabled=self.amp):
            (
                action_rot_x_one_hot,       # (b, 72)
                action_rot_y_one_hot,       # (b, 72)
                action_rot_z_one_hot,       # (b, 72)
                action_grip_one_hot,        # (b, 2)
                action_collision_one_hot,   # (b, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            rot_x_y = torch.cat(
                [action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ], dim=-1,)
            if self.rot_x_y_aug != 0:
                # add random interger between -2 and 2 to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape, device = rot_x_y.device)
                rot_x_y %= self._num_rotation_classes

            # t_end_2 = time.time()
            # if rank == 0:
            #     print("[DEBUG] Prepared inputs for network AVT_VGGT. Time Cost: {} minutes".format((t_end_2 - t_end_1) / 60.0))
            #     print("[DEBUG] Start timing for AVT_VGGT forward ...")

            out = self._network(
                pc=pc,                                                      # list of len b, (<4*128*128, 3)
                img_feat=img_feat,                                          # list of len b, (<4*128*128, 3)
                pixel_positions=pixel_positions,
                match_input_dict=match_input_dict,
                proprio=proprio,                                            
                lang_emb=lang_goal_embs,                                    
                img_aug=img_aug,                                            # 0
                wpt_local=wpt_local if self._network.training else None,    # (b, 3) float -1~1
                rot_x_y=rot_x_y,                                            # (b, 2) int 0~71
                rank=rank,
                intrinsics=train_intrinsics,
                extrinsics=train_extrinsics,
                iteration=step,
                epoch=epoch,
            )

            q_trans, rot_q, grip_q, collision_q = self.get_q(out, dims=(bs, nc, h, w))
            action_trans = self.get_action_trans(wpt_local, out, dims=(bs, nc, h, w), 
                                                 intrinsics=train_intrinsics, extrinsics=train_extrinsics)
            # # get task-related output
            # depth = out["depth"]
            # depth_conf = out["depth_conf"]
            # gt_depth = out["gt_depth"].clone()
            # valid_mask = out["valid_mask"]
            # if self.stage_two:
            #     depth_st = out["vggt2"]["depth"]
            #     depth_conf_st = out["vggt2"]["depth_conf"]
            #     gt_depth_st = out["vggt2"]["gt_depth2"].clone()
            #     valid_mask_st = out["vggt2"]["valid_mask2"]

        loss_log = {}
        if backprop:
            # t_start_loss = time.time()
            with autocast(device_type="cuda", enabled=self.amp):
                # cross-entropy loss
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()

                # # depth-related loss
                # beta_depth = 0
                # depth_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask, normalize_pred=False, 
                #                             normalize_gt=False, gamma=1.0, alpha=0.2, gradient_loss="grad", 
                #                             valid_range=-1.5, postfix="_depth")
                # depth_loss = depth_loss_dict['loss_conf_depth'] + depth_loss_dict['loss_grad_depth']
                # if self.stage_two:
                #     depth_loss_dict_st = conf_loss(depth_st, depth_conf_st, gt_depth_st, valid_mask_st, normalize_pred=False,
                #                                    normalize_gt=False, gamma=1.0, alpha=0.2, gradient_loss="grad", 
                #                                    valid_range=-10, postfix="_depth")
                #     depth_loss += depth_loss_dict_st['loss_conf_depth'] + depth_loss_dict_st['loss_grad_depth']
                ap_loss = self.calculate_matching_loss(out, vggt_features['point_map_view_1'], vggt_features['point_map_view_2'],
                                                       vggt_features['point_map_view_3'], vggt_features['point_map_view_4'], 
                                                       kp_1, kp_2, kp_3, kp_4, thres3d_neg = 0.1,) # --- Thresholds (can be tuned) ---

                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0

                rot_loss_x = self._cross_entropy_loss(
                    rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes,],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes,],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes,],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()

                grip_loss = self._cross_entropy_loss(
                    grip_q, action_grip_one_hot.argmax(-1),
                ).mean()

                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

                # 3D task-related total loss
                total_loss = (trans_loss + rot_loss_x + rot_loss_y + rot_loss_z + grip_loss + collision_loss + ap_loss)# + beta_depth * depth_loss
            # t_end_loss = time.time()
            # print("[DEBUG] loss calculation finished. Time Cost: {} minutes".format((t_end_loss - t_start_loss) / 60.0))
            
            self.scaler.scale(total_loss).backward()
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._optimizer.zero_grad(set_to_none=True)
            self._lr_sched.step()
            
            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                # "depth_loss": depth_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    rev_trans,
                    train_intrinsics,
                    train_extrinsics
                )

                return_log = manage_eval_log(
                    self = self,
                    tasks = tasks,
                    wpt = wpt,
                    pred_wpt = pred_wpt,
                    action_rot = action_rot,
                    pred_rot_quat = pred_rot_quat,
                    action_grip_one_hot = action_grip_one_hot,
                    grip_q = grip_q,
                    action_collision_one_hot = action_collision_one_hot,
                    collision_q = collision_q,
                    reset_log = reset_log,
                )

                return_out.update(return_log)

        return return_out
    
    # TODO: check this function
    @torch.no_grad()
    def act(
        self, step: int, rank, observation: dict, deterministic=True, 
    ) -> ActResult:

        lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
        _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
        lang_goal_embs = lang_goal_embs.float()
        proprio = stack_on_channel(observation["low_dim_state"])
        obs, pcd, eval_intrinsics, eval_extrinsics, original_rgbs = self.preprocess_images(observation, self.cameras) 
        eval_intrinsics = [k.float() for k in eval_intrinsics]  # float32
        eval_extrinsics = [e.float() for e in eval_extrinsics]
        original_rgbs = {k: v.float() for k, v in original_rgbs.items()}
        pc, img_feat, pixel_positions, colors = get_pc_img_feat_with_positions(obs, pcd, original_rgbs)
        pc, img_feat, pixel_positions, colors = move_pc_in_bound_with_positions(
            pc, img_feat, self.scene_bounds, colors, pixel_positions, no_op=not self.move_pc_in_bound
        )

        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = place_pc_in_cube(
                _pc,
                with_mean_or_bounds = False,
                scene_bounds = self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            pixel_positions=pixel_positions,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
            rank=rank,
            intrinsics=eval_intrinsics,
            extrinsics=eval_extrinsics,
            iteration=step
        ) # without wpt_local, rot_x_y, iteration; with different intrinsics and extrinsics

        _, rot_q, grip_q, collision_q = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, rev_trans, eval_intrinsics, eval_extrinsics
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        
        return ActResult(continuous_action)
    
    # TODO: check this function
    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        rev_trans,
        intrinsics, 
        extrinsics
    ):
        if self.stage_two:
            vggt1_or_vggt2 = False
            pred_wpt_local = self._net_mod.get_wpt(out, vggt1_or_vggt2)
        else:
            vggt1_or_vggt2 = True
            pred_wpt_local = self._net_mod.get_wpt(out, vggt1_or_vggt2, intrinsics, extrinsics)

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (rot_q[:, 0 * self._num_rotation_classes : 1 * self._num_rotation_classes,].argmax(1, keepdim=True),
             rot_q[:, 1 * self._num_rotation_classes : 2 * self._num_rotation_classes,].argmax(1, keepdim=True),
             rot_q[:, 2 * self._num_rotation_classes : 3 * self._num_rotation_classes,].argmax(1, keepdim=True),
            ), dim=-1,)
        pred_rot_quat = discrete_euler_to_quaternion(pred_rot.cpu(), self._rotation_resolution)
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    
    # TODO: check this function
    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        out,
        dims,
        intrinsics,
        extrinsics
    ):
        bs, nc, h, w = dims
        wpt_img = get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                vggt1_or_vggt2=False,
                dyn_cam_info=None,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc + 3

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans
    
    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()