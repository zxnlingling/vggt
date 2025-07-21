import os
import torch
import torch.nn.functional as F
from configs import vggt_config as vggt_cfg_mod
from configs import vggt_exp_config as exp_cfg_mod
from utils.mvt_utils import get_num_feat
from models import avt_vggt 
from models.avt_vggt_single import AVT_VGGT_Single
from einops import rearrange, repeat
from vggt_agent import VGGTAgent
from collections import defaultdict
from utils.env_utils import CAMERAS, SCENE_BOUNDS
from utils.vggt_utils import (get_pc_img_feat_with_positions,
                              move_pc_in_bound_with_positions)
from waypoint_extraction.select_keyframe import get_dataset
from utils.vggt_utils import (DATA_FOLDER, get_model_para, restore_cropped_image, get_depth, 
                              get_depth_st2, visualize_point_cloud, visualize_depth, get_pt_loc_on_img)
from utils.env_utils import COLOSSEUM_TASKS, RLBENCH_TASKS
from utils.mvt_utils import (stack_on_channel, apply_se3_aug_con, place_pc_in_cube, normalize_quaternion, discrete_euler_to_quaternion,
                             quaternion_to_discrete_euler, add_uni_noi, trans_pc, generate_hm_from_pt, select_feat_from_hm)
if hasattr(torch.amp, 'autocast'):
    # PyTorch>=2.3
    from torch.amp import autocast
else:
    from torch.cuda.amp import autocast

def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all_colosseum":
        tasks = COLOSSEUM_TASKS
        base_replay_dir = "/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/replay_train_varying_keypoints"
    elif parsed_tasks[0] == "all_rlbench":
        tasks = RLBENCH_TASKS
        base_replay_dir = "/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench"
    else:
        tasks = parsed_tasks
    return tasks, base_replay_dir

def preprocess_images(replay_sample, cameras): 
    obs, pcds = [], []
    intrinsics, extrinsics = [], []
    original_rgbs = {}
    use_input_pc = True
    for n in cameras:
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])             # torch.Size([B, 3, 128, 128]) 
        original_rgbs[n] = rgb.clone().detach()
        # point cloud
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])     # torch.Size([B, 3, 128, 128])
        rgb = (rgb.float() / 255.0) * 2.0 - 1.0                                      
        obs.append([rgb, pcd])
        pcds.append(pcd) 
        intrinsic = stack_on_channel(replay_sample["%s_camera_intrinsics" % n]) # (B, 3, 3)
        extrinsic = stack_on_channel(replay_sample["%s_camera_extrinsics" % n]) # (B, 4, 4)
        intrinsics.append(intrinsic)
        extrinsics.append(extrinsic)
    return obs, pcds, intrinsics, extrinsics, original_rgbs

def _get_one_hot_expert_actions(
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
        (bs, 72), dtype=int, device=device
    )
    action_rot_y_one_hot = torch.zeros(
        (bs, 72), dtype=int, device=device
    )
    action_rot_z_one_hot = torch.zeros(
        (bs, 72), dtype=int, device=device
    )
    action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
    action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

    # fill one-hots
    for b in range(bs):
        gt_rot = action_rot[b]
        gt_rot = quaternion_to_discrete_euler(
            gt_rot, 5
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
        depth_maps, valid_mask = get_depth(pixel_positions, pc)
        combined_feat = torch.cat([img, pc_img, depth_maps], dim=2)                 # (b, v, 3+3, H, W)
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
        # TODO: test point renderer in stage 1

    else: # stage 2
        # from renderers.rvt_renderer import RVTBoxRenderer
        # self.renderer = RVTBoxRenderer(device="cuda:2", img_size=(128, 128), 
        #                                                   three_views=True, with_depth=True,)
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=False):
                assert self.vggt1.add_corr and self.vggt1.norm_corr
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

        img = torch.cat(img, 0)
        # # 创建全黑图像并拼接
        # black_image = torch.zeros(
        #     img.shape[0],
        #     1,
        #     *img.shape[2:],
        #     device=img.device,
        #     dtype=img.dtype
        # )
        # img = torch.cat([img, black_image], dim=1)
        img = img.permute(0, 1, 4, 2, 3)
        depth_maps, valid_mask = get_depth_st2(img)

        # image augmentation
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        # assert pixel_positions is not None, "pixel_position must be provided in ray renderer"
        # img, pc_img = restore_cropped_image(img_feat, pixel_positions, points=pc)
        # dynamic_img = img[:,3]
        # bs = img.shape[0]
        # img_list = [img[i].unsqueeze(0) for i in range(bs)]
        # intrinsics_renderer = []
        # extrinsics_renderer = []
        # for i in range(bs):
        #     sample_intrinsics = torch.stack([K[i] for K in intrinsics])  # (4, 3, 3)
        #     sample_extrinsics = torch.stack([E[i] for E in extrinsics])  # (4, 4, 4)
        #     intrinsics_renderer.append(sample_intrinsics)
        #     extrinsics_renderer.append(sample_extrinsics)

        # with torch.no_grad():
        #     with autocast(device_type="cuda", enabled=False):
        #         assert self.vggt1.add_corr and self.vggt1.norm_corr
        #         img = []
        #         for _pc, _img_feat, _rgb_img, _intr, _extr in zip(pc, img_feat, img_list, 
        #                                                             intrinsics_renderer, extrinsics_renderer):
        #             # fix when the pc is empty
        #             max_pc = 1.0 if len(_pc) == 0 else torch.max(torch.abs(_pc))
        #             img.append(
        #                 self.renderer(
        #                     _pc,
        #                     torch.cat((_pc / max_pc, _img_feat), dim=-1),
        #                     input_images=_rgb_img,
        #                     intrinsics=_intr,
        #                     extrinsics=_extr,
        #                 ).unsqueeze(0)
        #             )

        # img = torch.cat(img, 0)
        # img = img.squeeze(1).permute(0, 2, 1, 3, 4)                        # (b, render_view_num, 7, 224, 224)
        # print("[DEBUG] img.shape= ", img.shape)
        # img = torch.cat([img, dynamic_img.unsqueeze(1)], dim=1)
        # print("[DEBUG] img.shape= ", img.shape)

        # # image augmentation
        # if img_aug != 0:
        #     stdv = img_aug * torch.rand(1, device=img.device)
        #     # values in [-stdv, stdv]
        #     noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
        #     img = torch.clamp(img + noise, -1, 1)

    if vggt.add_pixel_loc:
        bs = img.shape[0]
        pixel_loc = vggt.pixel_loc.to(img.device)
        img = torch.cat(
            (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
        )
        # Raw pixel_loc: -1.0 ~ 1.0
        # temp
        # img_base = torch.cat(
        #     (img_base, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
        # )

    return img, depth_maps, valid_mask


if __name__ == "__main__":
    rank = 2 
    device = f"cuda:{rank % 8}"

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    exp_cfg.bs = 5
    old_exp_cfg_peract_lr = exp_cfg.vggt.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    exp_cfg.vggt.lr *= 8 * exp_cfg.bs
    exp_cfg.freeze()

    EPOCHS = exp_cfg.epochs
    NUM_TRAIN = exp_cfg.demo
    BATCH_SIZE_TRAIN = exp_cfg.bs
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * 8))

    current_method = "base"
    waypoint_method = "heuristic" # fixed, heuristic, positional, geometric
    if waypoint_method == "geometric":
        replay_dir = "greedy_geometric"
        eval_log_dir = f"eval_geo/{current_method}_geo_vggt"
    elif waypoint_method == "positional":
        replay_dir = "dp_pos_gripper"
        eval_log_dir = f"eval_dp_pos/{current_method}_dp_pos_vggt"
    elif waypoint_method == "fixed":
        replay_dir = "fixed_number_10heuristic"
        eval_log_dir = f"eval_fixed_10/{current_method}_fixed_vggt"
    elif waypoint_method == "heuristic":
        replay_dir = "heuristic"
        eval_log_dir = f"eval_heuristic/{current_method}_heuristic_vggt"
    else:
        print("Unsupported waypoint method.")

    tasks, base_replay_dir = get_tasks(exp_cfg)
    TRAIN_REPLAY_STORAGE_DIR = os.path.join(base_replay_dir, replay_dir)
    log_dir_eval = os.path.join(base_replay_dir, eval_log_dir)

    vggt_cfg = vggt_cfg_mod.get_cfg_defaults()
    vggt_cfg.feat_dim = get_num_feat(exp_cfg.rvt)
    vggt_cfg.freeze()
    # for maintaining backward compatibility
    assert vggt_cfg.num_rot == exp_cfg.rvt.num_rotation_classes, print(
        vggt_cfg.num_rot, exp_cfg.rvt.num_rotation_classes
    )

    get_dataset_func = lambda: get_dataset(
        tasks=tasks,
        BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
        BATCH_SIZE_TEST=None,
        TRAIN_REPLAY_STORAGE_DIR=TRAIN_REPLAY_STORAGE_DIR,     # uncomment this line if training with RLBench         
        TEST_REPLAY_STORAGE_DIR=None,
        DATA_FOLDER=DATA_FOLDER,                            
        NUM_TRAIN=NUM_TRAIN,
        NUM_VAL=None,
        refresh_replay=False,
        device=device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,      
    )
    train_dataset = get_dataset_func()

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    vggt = avt_vggt.AVT_VGGT(
        renderer_device=device,
        rank=rank,
        **vggt_cfg,
    ).to(device)

    get_model_para(vggt)

    agent = VGGTAgent(
        network=vggt,
        cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        log_dir=f"/test_run/",
        **exp_cfg.rvt,
        **exp_cfg.vggt,
    )
    agent.build(training=True, device=device)

    start_epoch = 0
    end_epoch = EPOCHS

    if rank == 2:
        # logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.vggt.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.vggt.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        exp_cfg.vggt.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()

    i = start_epoch
    log_iter = 0
    dataset = train_dataset
    training_iterations = TRAINING_ITERATIONS 

    # ====== ready for train() ====== 

    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    raw_batch = next(data_iter)
    batch = {
        k: v.to(agent._device)
        for k, v in raw_batch.items()
        if type(v) == torch.Tensor
    }
    batch["tasks"] = raw_batch["tasks"]
    batch["lang_goal"] = raw_batch["lang_goal"]
    
    step = 0 # iteration
    replay_sample = batch
    backprop = True

    # ====== ready for agent update() ====== 

    assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
    assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
    assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
    assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
    assert replay_sample["low_dim_state"].shape[1:] == (1, vggt.proprio_dim,)

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

    obs, pcd, intrinsics, extrinsics, original_rgbs = preprocess_images(replay_sample, CAMERAS)    
    transform_augmentation_rpy = [0.0, 0.0, 45.0]
    transform_augmentation_xyz = [0.125, 0.125, 0.125]   
        
    with torch.no_grad():
        # get pc and rgb features of shape (len b, H * W * 4, 3)
        pc, img_feat, pixel_positions, colors = get_pc_img_feat_with_positions(obs, pcd, original_rgbs)

        # apply SE3 augmentation to a point clouds and actions
        action_trans_con, action_rot, pc = apply_se3_aug_con(
            pcd=pc,
            action_gripper_pose=action_gripper_pose,
            bounds=torch.tensor(SCENE_BOUNDS),
            trans_aug_range = torch.tensor(transform_augmentation_xyz),
            rot_aug_range=torch.tensor(transform_augmentation_rpy),
        )
        action_trans_con = torch.tensor(action_trans_con).to(pc.device)
        action_rot = torch.tensor(action_rot).to(pc.device)

        action_rot = action_rot.cpu().numpy()
        for i, _action_rot in enumerate(action_rot):
            _action_rot = normalize_quaternion(_action_rot)
            if _action_rot[-1] < 0:
                _action_rot = -_action_rot
            action_rot[i] = _action_rot

        pc, img_feat, pixel_positions, _ = move_pc_in_bound_with_positions(
            pc, img_feat, SCENE_BOUNDS, colors, pixel_positions, no_op=False
        )
        wpt = [x[:3] for x in action_trans_con]

        wpt_local = []
        rev_trans = []
        for _pc, _wpt in zip(pc, wpt):
            a, b = place_pc_in_cube(_pc, _wpt, with_mean_or_bounds=False, scene_bounds=SCENE_BOUNDS)
            wpt_local.append(a.unsqueeze(0))
            rev_trans.append(b)

        wpt_local = torch.cat(wpt_local, axis=0)
        pc = [place_pc_in_cube(_pc, with_mean_or_bounds = False, scene_bounds = SCENE_BOUNDS,)[0] for _pc in pc]
        bs = len(pc)
        nc = vggt.num_img
        h = w = vggt.img_size
        img_aug = 0

    with autocast(device_type="cuda", enabled=True):
        (
            action_rot_x_one_hot,       # (b, 72)
            action_rot_y_one_hot,       # (b, 72)
            action_rot_z_one_hot,       # (b, 72)
            action_grip_one_hot,        # (b, 2)
            action_collision_one_hot,   # (b, 2)
        ) = _get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=device
        )

        rot_x_y = torch.cat(
            [action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
            ], dim=-1,)

        # add random interger between -2 and 2 to rot_x_y
        rot_x_y += torch.randint(
            -2, 2, size=rot_x_y.shape
        ).to(rot_x_y.device)
        rot_x_y %= 72
    
    lang_emb=lang_goal_embs
    iteration=step

    # ====== ready to input to avt_vggt.AVT_VGGT ====== 

    with torch.no_grad():
        for x in img_feat:
            stdv = 0.05 * torch.rand(1, device=x.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
            x = x + noise # TODO: the noise is not added to img_feat
        # (b, 3, 10, 224, 224) or (b, 4, 9, 128, 128)   
        vggt.use_renderer = False  
        img, depth_maps, valid_mask = render(self=vggt, pc=pc, img_feat=img_feat, img_aug=img_aug, 
                     vggt1_or_vggt2=True, pixel_positions=pixel_positions)

    wpt_local_stage_one = wpt_local
    wpt_local_stage_one = wpt_local_stage_one.clone().detach()
    wpt_local=wpt_local_stage_one

    # ====== ready to input to AVT_VGGT_Single (no_feat=self.stage_two) ======
    # no_feat = True
    # bs, num_img, img_feat_dim, h, w = img.shape         
    # img_raw = img.clone()

    # use VGGT to encode rgb_img
    # rgb_img = img_raw[:, :, :3, :, :]
    
    # # 检查 RGB
    # from utils.vggt_utils import save_rgb_images
    # # save_rgb_images(rgb_img, "debug_runs/images/temp_stage1", prefix="original", views=CAMERAS)
    # save_rgb_images(img[:, :, :3, :, :], "debug_runs/images/temp_stage2/st1", prefix="original", views=CAMERAS)
    # # save_rgb_images(img[:, :, 3:6, :, :], "debug_runs/images/temp_stage1/baseline_imgs", prefix="original", views=CAMERAS)
    # rgb_img = img[:, :, :3, :, :]
 
    # # resize for VGGT
    # if rgb_img.shape[-1] != vggt.vggt1.image_resolution:
    #     original_shape = rgb_img.shape
    #     # Reshape to (B*V, 3, H, W)
    #     rgb_img = rgb_img.view(-1, *rgb_img.shape[2:])
    #     resized_img = F.interpolate(rgb_img, size=(vggt.vggt1.image_resolution, vggt.vggt1.image_resolution), 
    #                                 mode='bicubic', align_corners=False)
    #     resized_img = torch.clamp(resized_img, 0.0, 1.0)
    #     # Restore to (B, V, 3, H, W)
    #     resized_img = resized_img.view(*original_shape[:2], *resized_img.shape[1:])
    #     rgb_img = resized_img               # (b, render_view_num, 3, 182, 182)  
    
    # with torch.cuda.amp.autocast(enabled=True):
    
    # #     vggt.vggt1.vggt_rgb_feats_all = [None] * 5

    #     vggt_rgb_feats, vggt_feats_16, vggt_feats_32, vggt_feats_64, vggt_feats_128, depth, depth_conf = \
    #         vggt.vggt1.vggt_image_encoder_forward(vggt.vggt1.vggt, rgb_img, rank)    
    #     height, width = vggt_rgb_feats.shape[-2:]  
    #     vggt.vggt1.vggt_rgb_feats_all[0] = vggt_feats_16
    #     vggt.vggt1.vggt_rgb_feats_all[1] = vggt_feats_32
    #     vggt.vggt1.vggt_rgb_feats_all[2] = vggt_feats_64
    #     vggt.vggt1.vggt_rgb_feats_all[3] = vggt_feats_128
    #     vggt.vggt1.vggt_rgb_feats_all[-1] = vggt_rgb_feats.view(bs*num_img, vggt.vggt1.vggt_feat_dim, height, width)

    #     vggt_out = vggt.vggt1.vggt_rgb_feats_all[-1]  

    # # c 128 -> vggt_img_dim
    # rgb_img = vggt.vggt1.fusion(vggt_out)
    # num_pat_img = h // vggt.vggt1.img_patch_size
    # rgb_img = (rgb_img.view(bs, num_img, vggt.vggt1.vggt_img_dim, num_pat_img, num_pat_img).transpose(1, 2).clone()) # (b, vggt_img_dim, v, 16, 16) 

    # normalize RGB to [-1, 1]
    # img_rgb_normalized = (img_raw[:, :, :3] - 0.5) * 2
    # img_combined = torch.cat([img_rgb_normalized, img_raw[:, :, 3:]], dim=2)

    # # 检查点云
    # import matplotlib.pyplot as plt
    # pc_temp = img[:, :, 3:6, :, :].cpu().numpy()

    # save_dir_pc = "debug_runs/images/temp_stage2/st1/point_clouds"
    # os.makedirs(save_dir_pc, exist_ok=True)

    # # 保存3D点云（原始点云）
    # for i in range(min(5, len(pc))):  # 保存前5个样本
    #     sample_pc = pc[i].cpu().numpy()
    #     save_path = os.path.join(save_dir_pc, f"original_sample{i}.png")
    #     visualize_point_cloud(sample_pc, title=f"Original PC Sample {i}", save_path=save_path)

    # # 保存2D点云投影（恢复后的点云）
    # for i in range(min(5, pc_temp.shape[0])):  # 保存前5个样本
    #     for view_idx in range(min(4, pc_temp.shape[1])):  # 保存前4个视角
    #         # 转换为 [h, w, 3]
    #         pc_restore = pc_temp[i, view_idx].transpose(1, 2, 0)
    #         plt.imshow((pc_restore * 0.5 + 0.5).clip(0, 1))  # 假设归一化到 [-1, 1]
    #         plt.title(f"Restored PC Sample {i}, View {view_idx}")
    #         plt.axis('off')  # 关闭坐标轴
    #         save_path = os.path.join(save_dir_pc, f"sample{i}_view{view_idx}.png")
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #         plt.close()

    # # temp test for depth
    # from utils.vggt_utils import save_depth_images
    # save_depth_dir = "debug_runs/images/temp_stage2/st1/depth"
    # save_depth_images(img[:, :, 6, :, :], save_depth_dir, 0, iteration, prefix="GT", 
    #                           views=["front", "left", "right", "wrist"])

    # depth_orig = img[:, :, 6, :, :].unsqueeze(2)  # (bs, c, 1, h, w)
    # pc_orig = img[:, :, 3:6, :, :]            # (bs, c, 3, h, w)
    # rgb_orig = img[:, :, :3, :, :]           # (bs, c, 3, h, w)
    # combined_orig = torch.cat([rgb_orig, pc_orig, depth_orig, img[:, :, 7:10, :, :]], dim=2)
    # is_orig_combined_valid = torch.equal(combined_orig, img)
    # # print("[DEBUG] pixel_loc check: ", img_base[:, :, 6:9, :, :] == img[:, :, 7:10, :, :])
    # # print("[DEBUG] dimension check: orig render", img.shape, img.shape)
    # print("[DEBUG] depth check st1:", depth_orig.shape, depth_orig.min(), depth_orig.max())
    # print("[DEBUG] pc check st1:", pc_orig.shape, pc_orig.min(), pc_orig.max())
    # print("[DEBUG] rgb check st1:", rgb_orig.shape, rgb_orig.min(), rgb_orig.max())
    # print("[DEBUG] division check st1:", is_orig_combined_valid, combined_orig.shape)

    # feat_img = img_combined.view(bs * num_img, img_feat_dim, h, w)

    # feat_img = vggt.vggt1.patchify(feat_img)                  # Conv2DBlock   (b*v, feat_img_dim/2, 16, 16)

    # if vggt.vggt1.add_depth:
    #     # visualize depth and original rgb
    #     visualize_depth(resized_img, depth, depth_conf, 0, iteration, save_dir=save_depth_dir)
    #     # add depth to feat_img for the first stage
    #     depth = depth.permute(0, 1, 4, 2, 3).view(bs * num_img, 1, height, width)
    #     depth = 2 * (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) - 1
    #     depth_conf = depth_conf.unsqueeze(2).view(bs * num_img, 1, height, width)
    #     depth_conf = torch.sigmoid(depth_conf)
    #     depth_weighted = depth * depth_conf  # (b*v, 1, h, w)

    #     # import matplotlib.pyplot as plt
    #     # depth_vis = depth_weighted.view(bs, num_img, height, width).cpu().detach()
    #     # save_dir = "debug_runs/images/temp_stage1/depth_weighted_maps"
    #     # os.makedirs(save_dir, exist_ok=True)
    #     # for i in range(bs):  # 样本数量
    #     #     for j in range(num_img):  # 视角数量
    #     #         plt.figure(figsize=(6, 6))
    #     #         plt.imshow(depth_vis[i, j].numpy(), cmap='viridis')  # 使用 'viridis' 或 'plasma' 颜色映射
    #     #         plt.colorbar()
    #     #         plt.title(f"Sample {i}, View {j} Depth")
    #     #         plt.axis('off')
    #     #         file_path = os.path.join(save_dir, f"sample_{i}_view_{j}.png")
    #     #         plt.savefig(file_path, dpi=300, bbox_inches='tight')
    #     #         plt.close()

    #     feat_depth = vggt.vggt1.depth_patchify(depth_weighted)         # (b*v, feat_img_dim/2, 16, 16)
    #     feat_img = torch.cat([feat_img, feat_depth], dim=1)
        
    # feat_img = (
    #     feat_img.view(
    #         bs,
    #         num_img,
    #         vggt.vggt1.feat_img_dim,
    #         num_pat_img,
    #         num_pat_img,
    #     ).transpose(1, 2).clone())          
    # _, _, _d, _h, _w = feat_img.shape


    
    wpt_img4trans = get_pt_loc_on_img(
        wpt_local.unsqueeze(1),
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    wpt_img4trans = wpt_img4trans.squeeze(1)
    action_trans_1 = generate_hm_from_pt(
        wpt_img4trans.reshape(-1, 2),
        (h, w),
        sigma=1.5,
        thres_sigma_times=3,
    )
    GT_trans_1 = action_trans_1.view(bs, nc, h * w).transpose(1, 2).clone()
    trans_temp_1 = GT_trans_1.contiguous().view(bs, 4, 1, h, w).half().to(device)

    out = {
        "trans": trans_temp_1,
        }
    


    # ====== prepare input to AVT_VGGT_Single (no_feat=False) of Stage 2 ======
    
    with torch.no_grad():
        pc_temp = pc
        wpt_local_stage_one_noisy = add_uni_noi(        # (b, 3)
            wpt_local_stage_one.clone().detach(), 2 * 0.05
        )
        pc, rev_trans_st2 = trans_pc(
            pc, loc=wpt_local_stage_one_noisy, sca=4
        )   
        wpt_local2, _ = trans_pc(                       # (b, 3)
            wpt_local, loc=wpt_local_stage_one_noisy, sca=4
        )

        wpt_local1_eval = vggt.get_wpt(out, vggt1_or_vggt2=True, intrinsics=intrinsics, extrinsics=extrinsics)
        pc_eval, rev_trans_st2_eval = trans_pc(pc_temp, loc=wpt_local, sca=4)

        img, depth_maps_st, valid_mask_st = render(
            self=vggt, 
            pc=pc,
            img_feat=img_feat,
            img_aug=img_aug,
            vggt1_or_vggt2=False
        )
    out.update({"wpt_local1": wpt_local_stage_one_noisy,
                "rev_trans": rev_trans_st2,})
    
    # ====== ready to input to AVT_VGGT_Single (no_feat=False) ======
        # rgb_img = img[:, :, 3:6, :, :]
    # # resize for VGGT
    # if rgb_img.shape[-1] != vggt.vggt1.image_resolution:
    #     original_shape = rgb_img.shape
    #     # Reshape to (B*V, 3, H, W)
    #     rgb_img = rgb_img.view(-1, *rgb_img.shape[2:])
    #     resized_img = F.interpolate(rgb_img, size=(vggt.vggt2.image_resolution, vggt.vggt2.image_resolution), 
    #                                 mode='bicubic', align_corners=False)
    #     resized_img = torch.clamp(resized_img, 0.0, 1.0)
    #     # Restore to (B, V, 3, H, W)
    #     resized_img = resized_img.view(*original_shape[:2], *resized_img.shape[1:])
    #     rgb_img = resized_img               # (b, render_view_num, 3, 182, 182)  
    
    # with torch.cuda.amp.autocast(enabled=True):
    
    # #     vggt.vggt1.vggt_rgb_feats_all = [None] * 5

    #     vggt_rgb_feats, vggt_feats_16, vggt_feats_32, vggt_feats_64, vggt_feats_128, depth, depth_conf = \
    #         vggt.vggt2.vggt_image_encoder_forward(vggt.vggt2.vggt, rgb_img, rank)
        
    # # 检查 RGB
    # from utils.vggt_utils import save_rgb_images
    # save_rgb_images(img[:, :, 3:6, :, :], "debug_runs/images/temp_stage2/st2", prefix="original", views=CAMERAS)
    # # 检查点云
    # import matplotlib.pyplot as plt
    # pc_temp = img[:, :, :3, :, :].cpu().numpy()

    # save_dir_pc = "debug_runs/images/temp_stage2/st2/point_clouds"
    # os.makedirs(save_dir_pc, exist_ok=True)

    # # 保存3D点云（原始点云）
    # for i in range(min(5, len(pc))):  # 保存前5个样本
    #     sample_pc = pc[i].cpu().numpy()
    #     save_path = os.path.join(save_dir_pc, f"original_sample{i}.png")
    #     visualize_point_cloud(sample_pc, title=f"Original PC Sample {i}", save_path=save_path)

    # # 保存2D点云投影（恢复后的点云）
    # for i in range(min(5, pc_temp.shape[0])):  # 保存前5个样本
    #     for view_idx in range(min(4, pc_temp.shape[1])):  # 保存前4个视角
    #         # 转换为 [h, w, 3]
    #         pc_restore = pc_temp[i, view_idx].transpose(1, 2, 0)
    #         plt.imshow((pc_restore * 0.5 + 0.5).clip(0, 1))  # 假设归一化到 [-1, 1]
    #         plt.title(f"Restored PC Sample {i}, View {view_idx}")
    #         plt.axis('off')  # 关闭坐标轴
    #         save_path = os.path.join(save_dir_pc, f"sample{i}_view{view_idx}.png")
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #         plt.close()

    # # temp test for depth
    # from utils.vggt_utils import save_depth_images
    # save_depth_dir = "debug_runs/images/temp_stage2/st2/depth"
    # save_depth_images(img[:, :, 6, :, :], save_depth_dir, 0, iteration, prefix="GT", 
    #                           views=["front", "left", "right", "wrist"])

    # depth_st = img[:, :, 6, :, :].unsqueeze(2)  # (bs, c, 1, h, w)
    # rgb_st = img[:, :, 3:6, :, :]            # (bs, c, 3, h, w)
    # pc_st = img[:, :, :3, :, :]           # (bs, c, 3, h, w)
    # combined_st = torch.cat([pc_st, rgb_st, depth_st, img[:, :, 7:10, :, :]], dim=2)
    # is_st_combined_valid = torch.equal(combined_st, img)
    # # print("[DEBUG] pixel_loc check: ", img_base[:, :, 6:9, :, :] == img[:, :, 7:10, :, :])
    # # print("[DEBUG] dimension check: orig render", img.shape, img.shape)
    # print("[DEBUG] depth check st2:", depth_st.shape, depth_st.min(), depth_st.max())
    # print("[DEBUG] pc check st2:", pc_st.shape, pc_st.min(), pc_st.max())
    # print("[DEBUG] rgb check st2:", rgb_st.shape, rgb_st.min(), rgb_st.max())
    # print("[DEBUG] division check st2:", is_st_combined_valid, combined_st.shape)

    # if vggt.vggt2.add_depth:
    #     # visualize depth and original rgb
    #     visualize_depth(resized_img, depth, depth_conf, 0, iteration, save_dir=save_depth_dir)
    

    # out_vggt2 = self.vggt2(
    #     img=img,
    #     # proprio=proprio,
    #     # lang_emb=lang_emb,
    #     wpt_local=wpt_local2,
    #     # rot_x_y=rot_x_y,
    #     # rank=rank,
    #     intrinsics=intrinsics,
    #     extrinsics=extrinsics,
    #     rev_trans=rev_trans_st2,
    #     **kwargs,
    # )
    x_temp0 = torch.rand(bs, vggt.vggt2.input_dim_before_seq, 3, 16, 16).to(device)
    feat = []
    _feat = torch.max(torch.max(x_temp0, dim=-1)[0], dim=-1)[0]               # for global max of x
    _feat = rearrange(_feat, 'b c n -> b (c n)')                             
    feat.append(repeat(_feat, f'b d -> b {1} d')) 
    x_temp = (                                                               # (b*v, 128, 16, 16)
            x_temp0.transpose(1, 2)
            .contiguous()
            .view(
                bs * vggt.vggt2.num_img, vggt.vggt2.input_dim_before_seq, 16, 16
            )
        ).float().to(device)
    



    wpt_img2trans = vggt.get_pt_loc_on_img(wpt_local.unsqueeze(1), vggt1_or_vggt2=False, dyn_cam_info=None, out=out,)
    # wpt_img2trans = wpt_img2trans.squeeze(1)
    action_trans_2 = generate_hm_from_pt(
        wpt_img2trans.reshape(-1, 2),
        (h, w),
        sigma=1.5,
        thres_sigma_times=3,
    )
    GT_trans_2 = action_trans_2.view(bs, 3, h * w).transpose(1, 2).clone()
    trans_temp_2 = GT_trans_2.contiguous().view(bs, 3, 1, h, w).half().to(device)


    wpt_img = vggt.vggt2.renderer.get_pt_loc_on_img(
        wpt_local2.unsqueeze(1), fix_cam=True, dyn_cam_info=None
    )
    wpt_img = wpt_img.reshape(bs * vggt.vggt2.num_img, 2)
    wpt_img = add_uni_noi(
        wpt_img, vggt.vggt2.wpt_img_aug * vggt.vggt2.img_size
    )
    wpt_img = torch.clamp(wpt_img, 0, vggt.vggt2.img_size - 1)
    _wpt_img = wpt_img / vggt.vggt2.img_patch_size
    _u = x_temp
    assert (
        0 <= _wpt_img.min() and _wpt_img.max() <= x_temp.shape[-1]
    ), print(_wpt_img, x_temp.shape)
    _wpt_img = _wpt_img.unsqueeze(1)                               # (b*4, 1, 2)
    _feat = select_feat_from_hm(_wpt_img, _u)[0]
    _feat = _feat.view(bs, 1, -1) 
    feat.append(_feat)
    feat = torch.cat(feat, dim=-1)
    feat = feat.squeeze(1)

    feat_ex_rot = vggt.vggt2.feat_fc_ex_rot(feat)
    feat_rot = vggt.vggt2.feat_fc_init_bn(feat)
    feat_x = vggt.vggt2.feat_fc_x(feat_rot)
    rot_x = rot_x_y[..., 0].view(bs, 1)
    rot_x_pe = vggt.vggt2.feat_fc_pe(rot_x)
    feat_y = vggt.vggt2.feat_fc_y(feat_rot + rot_x_pe)
    rot_y = rot_x_y[..., 1].view(bs, 1)
    rot_y_pe = vggt.vggt2.feat_fc_pe(rot_y)
    feat_z = vggt.vggt2.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)

    
    out2 = {
        "trans": trans_temp_2,
        "feat_x": feat_x.unsqueeze(1),
        "feat_y": feat_y.unsqueeze(1),
        "feat_z": feat_z.unsqueeze(1),
        "feat_ex_rot": feat_ex_rot.unsqueeze(1),
    }
    out["vggt2"] = out2







    q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
    q_trans = q_trans.clone()
    out2 = out["vggt2"]
    q_trans2 = out2["trans"].view(bs, 3, h * w).transpose(1, 2)
    q_trans2 = q_trans2.clone()
    q_trans = torch.cat((q_trans, q_trans2), dim=2)
    rot_q = torch.cat((out2["feat_x"], out2["feat_y"], out2["feat_z"]), dim=-1).view(bs, -1)
    grip_q = out2["feat_ex_rot"].view(bs, -1)[:, :2]
    collision_q = out2["feat_ex_rot"].view(bs, -1)[:, 2:]


    # get ground truth
    wpt_img = get_pt_loc_on_img(
        wpt_local.unsqueeze(1),
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    assert wpt_img.shape[1] == 1
    wpt_img2 = vggt.get_pt_loc_on_img(
        wpt_local.unsqueeze(1),
        vggt1_or_vggt2=False,
        dyn_cam_info=None,
        out=out,
    )
    assert wpt_img2.shape[1] == 1
    wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
    nc = nc + 3
    wpt_img = wpt_img.squeeze(1)
    action_trans = generate_hm_from_pt(
        wpt_img.reshape(-1, 2),
        (h, w),
        sigma=1.5,
        thres_sigma_times=3,
    )
    action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()


    
    # ===== loss while training
    print(f"Train error - trans: {torch.abs(q_trans - action_trans).mean().item()}")
    print(f"Train error - rot_x: {F.cross_entropy(rot_q[:, 0 * 72 : 1 * 72,], action_rot_x_one_hot.argmax(-1)).mean().item()}")
    print(f"Train error - rot_y: {F.cross_entropy(rot_q[:, 1 * 72 : 2 * 72,], action_rot_y_one_hot.argmax(-1)).mean().item()}")
    print(f"Train error - rot_z: {F.cross_entropy(rot_q[:, 2 * 72 : 3 * 72,], action_rot_z_one_hot.argmax(-1)).mean().item()}")
    print(f"Train error - grip: {torch.abs(grip_q - action_grip_one_hot).mean().item()}")
    print(f"Train error - collision: {torch.abs(collision_q - action_collision_one_hot).mean().item()}")  

    # from utils.vggt_utils import conf_loss, resize_gt_depth
    # gt_depth, valid_mask = resize_gt_depth(depth_maps, valid_mask, size=vggt.image_resolution)

    # loss_log = {}
    # with autocast(device_type="cuda", enabled=True):
    #     # depth-related loss
    #     depth_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask, normalize_pred=False, 
    #                                 normalize_gt=False, gamma=1.0, alpha=0.2, gradient_loss="grad", 
    #                                 valid_range=-1.5, postfix="_depth")
    #     print(depth_loss_dict)
    #     depth_loss = depth_loss_dict['loss_conf_depth'] + depth_loss_dict['loss_grad_depth']




# act ###############################################################################################
    feat = []
    _feat = torch.max(torch.max(x_temp0, dim=-1)[0], dim=-1)[0]               # for global max of x
    _feat = rearrange(_feat, 'b c n -> b (c n)')                             
    feat.append(repeat(_feat, f'b d -> b {1} d')) 
    x_temp = (                                                               # (b*v, 128, 16, 16)
            x_temp0.transpose(1, 2)
            .contiguous()
            .view(
                bs * vggt.vggt2.num_img, vggt.vggt2.input_dim_before_seq, 16, 16
            )
        ).float().to(device)
    





#  wpt_local = torch.tensor([[ 0.4521, -0.2625, -0.4767], [ 0.1640, -0.3012, -0.0685], 
#                            [-0.1202, -0.8019, -0.0326], [ 0.4682, -0.4218, -0.1900], 
#                           [-2.6523e-01,  7.3418e-05, -4.8443e-01]], device=device)
#  wpt_local_stage_one_noisy = add_uni_noi(wpt_local.clone().detach(), 2 * 0.05)  
# wpt_local2, _ = trans_pc(wpt_local, loc=wpt_local_stage_one_noisy, sca=4)
# tensor([[-0.1731,  0.0410, -0.1392]], device='cuda:7')  tensor([[ 0.3114, -0.2718,  0.1823]], device='cuda:7') tensor([[-0.0030,  0.1614, -0.2086]], device='cuda:7')
# tensor([[-0.2484, -0.3402,  0.2556]], device='cuda:7')  tensor([[-0.3955,  0.0665, -0.0396]], device='cuda:7')





    # TODO: wrong use of get_wpt    wpt_local2_eval → wpt_local2
    wpt_local2_eval = vggt.vggt2.get_wpt(out={"trans": trans_temp_2.clone().detach()}, vggt1_or_vggt2=False)
    print(wpt_local2.shape)
    print(f"wpt_local2_eval = {wpt_local2_eval}, wpt_local2 = {wpt_local2}, gap = \
          {torch.abs(wpt_local2 - wpt_local2_eval).mean().item()}")

    wpt_img = vggt.vggt2.renderer.get_pt_loc_on_img(wpt_local2_eval.unsqueeze(1), fix_cam=True, dyn_cam_info=None)
    wpt_img = wpt_img.reshape(bs * vggt.vggt2.num_img, 2)
    wpt_img = torch.clamp(wpt_img, 0, vggt.vggt2.img_size - 1)
    _wpt_img = wpt_img / vggt.vggt2.img_patch_size
    _u = x_temp
    assert (
        0 <= _wpt_img.min() and _wpt_img.max() <= x_temp.shape[-1]
    ), print(_wpt_img, x_temp.shape)
    _wpt_img = _wpt_img.unsqueeze(1)                               # (b*4, 1, 2)
    _feat = select_feat_from_hm(_wpt_img, _u)[0]
    _feat = _feat.view(bs, 1, -1) 
    feat.append(_feat)
    feat = torch.cat(feat, dim=-1)
    feat = feat.squeeze(1)

    feat_ex_rot = vggt.vggt2.feat_fc_ex_rot(feat)
    feat_rot = vggt.vggt2.feat_fc_init_bn(feat)
    feat_x = vggt.vggt2.feat_fc_x(feat_rot)
    rot_x = feat_x.argmax(dim=1, keepdim=True)
    rot_x_pe = vggt.vggt2.feat_fc_pe(rot_x)
    feat_y = vggt.vggt2.feat_fc_y(feat_rot + rot_x_pe)
    rot_y = feat_y.argmax(dim=1, keepdim=True)
    rot_y_pe = vggt.vggt2.feat_fc_pe(rot_y)
    feat_z = vggt.vggt2.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)

    out_eval = {
        "trans": trans_temp_1,
        }
    out_eval.update({"wpt_local1": wpt_local1_eval,
                     "rev_trans": rev_trans_st2_eval})
    out_eval2 = {
        "trans": trans_temp_2,
        "feat_x": feat_x.unsqueeze(1),
        "feat_y": feat_y.unsqueeze(1),
        "feat_z": feat_z.unsqueeze(1),
        "feat_ex_rot": feat_ex_rot.unsqueeze(1),
    }
    out_eval["vggt2"] = out_eval2



    rot_q_eval = torch.cat((out_eval["vggt2"]["feat_x"], out_eval["vggt2"]["feat_y"], out_eval["vggt2"]["feat_z"]), dim=-1).view(bs, -1)
    grip_q_eval = out_eval["vggt2"]["feat_ex_rot"].view(bs, -1)[:, :2]
    collision_q_eval = out_eval["vggt2"]["feat_ex_rot"].view(bs, -1)[:, 2:]

    vggt1_or_vggt2 = False
    pred_wpt_local = vggt.get_wpt(out_eval, vggt1_or_vggt2)
    # wpt = self.vggt2.get_wpt(out["vggt2"])
    # hm = F.softmax(out_eval["vggt2"]["trans"].view(bs, nc, h * w) / 0.005, 2).view(bs, nc, h, w)
    # wpt = [self.renderer.get_max_3d_frm_hm_cube(hm[i : i + 1], fix_cam=True, dyn_cam_info=None,) for i in range(bs)]
    # wpt = out_eval["rev_trans"](wpt)
    pred_wpt = []
    for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
        pred_wpt.append(_rev_trans(_pred_wpt_local))
    pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])
    

    pred_rot = torch.cat(
        (rot_q_eval[:, 0 * 72 : 1 * 72,].argmax(1, keepdim=True),
         rot_q_eval[:, 1 * 72 : 2 * 72,].argmax(1, keepdim=True),
         rot_q_eval[:, 2 * 72 : 3 * 72,].argmax(1, keepdim=True),
        ), dim=-1,)
    pred_rot_quat = discrete_euler_to_quaternion(pred_rot.cpu(), 5)
    pred_grip = grip_q_eval.argmax(1, keepdim=True)         # only caused by wpt_local between train and eval
    pred_coll = collision_q_eval.argmax(1, keepdim=True)    # only caused by wpt_local between train and eval


    # ===== eval gap
    print(f"[DEBUG] wpt_local = {wpt_local}, pred_wpt_local = {pred_wpt_local}")
    print(f"Eval gap - trans: {torch.abs(wpt_local - pred_wpt_local).mean().item()}")
    print(f"true gap - trans: {torch.abs(action_gripper_pose[:, 0:3] - pred_wpt).mean().item()}")
    print(f"Eval gap - rot_x: {torch.abs(rot_q[:, 0 * 72 : 1 * 72,] - rot_q_eval[:, 0 * 72 : 1 * 72,]).mean().item()}")
    print(f"Eval gap - rot_y: {torch.abs(rot_q[:, 1 * 72 : 2 * 72,] - rot_q_eval[:, 1 * 72 : 2 * 72,]).mean().item()}")
    print(f"Eval gap - rot_z: {torch.abs(rot_q[:, 2 * 72 : 3 * 72,] - rot_q_eval[:, 2 * 72 : 3 * 72,]).mean().item()}")
    print(f"true gap - rot: {torch.abs(action_gripper_pose[:, 3:7].cpu() - pred_rot_quat).mean().item()}")

    # rot_y caused by wpt_local and rot_x between train and eval
    # rot_z caused by wpt_local and rot_x and rot_y between train and eval

    print(f"Eval gap - grip: {torch.abs(grip_q - grip_q_eval).mean().item()}")
    print(f"Eval gap - collision: {torch.abs(collision_q - collision_q_eval).mean().item()}") 
    print(f"true gap - grip: {torch.abs(action_grip.float() - pred_grip).mean().item()}")
    print(f"true gap - collision: {torch.abs(action_ignore_collisions.float() - pred_coll).mean().item()}")