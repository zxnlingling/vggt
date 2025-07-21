import torch
# from transformers import AutoTokenizer, AutoModel, AutoConfig

# config = AutoConfig.from_pretrained("./Sa2VA-4B", trust_remote_code=True)
# print(config)
# from PIL import Image
import numpy as np
import os

# # load the model and tokenizer
# path = "./Sa2VA-4B"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True).eval().cuda()
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# # ===== for image chat
# # text_prompts = "<image>Please describe the image."

# # ===== for image chat with segmentation output
# image_path = "/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench/train/close_jar/all_variations/episodes/episode0/front_rgb/0.png"
# text_prompts = "<image>Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer."
# image = Image.open(image_path).convert('RGB')
# input_dict = {
#     'image': image,
#     'text': text_prompts,
#     'past_text': '',
#     'mask_prompts': None,
#     'tokenizer': tokenizer,
#     }
# return_dict = model.predict_forward(**input_dict)
# answer = return_dict["prediction"] # the text format answer
# masks = return_dict['prediction_masks']  # segmentation masks, list(np.array(1, h, w), ...)
    
# # ===== for chat with visual prompt (mask format) input
# # mask_prompts = np.load('/PATH/TO/pred_masks.npy') # np.array(n_prompts, h, w)
# # image_path = "/PATH/TO/IMAGE"
# # text_prompts = "<image>Can you provide me with a detailed description of the region in the picture marked by region1."
# # image = Image.open(image_path).convert('RGB')
# # input_dict = {
# #     'image': image,
# #     'text': text_prompts,
# #     'past_text': '',
# #     'mask_prompts': mask_prompts,
# #     'tokenizer': tokenizer,
# #     }
# # return_dict = model.predict_forward(**input_dict)
# # answer = return_dict["prediction"] # the text format answer

# # ===== for video chat
# # if len(images_paths) > 5:  # uniformly sample 5 frames
# #     step = (len(images_paths) - 1) // (5 - 1)
# #     images_paths = [images_paths[0]] + images_paths[1:-1][::step][1:] + [images_paths[-1]]
# # text_prompts = "<image>Please describe the video."


# # ===== for video chat with segmentation mask output
# video_folder = "/PATH/TO/VIDEO_FOLDER"
# images_paths = os.listdir(video_folder)
# images_paths = [os.path.join(video_folder, image_path) for image_name in images_paths]
# text_prompts = "<image>Please segment the person."
# input_dict = {
#     'video': images_paths,
#     'text': text_prompts,
#     'past_text': '',
#     'mask_prompts': None,
#     'tokenizer': tokenizer,
# }
# return_dict = model.predict_forward(**input_dict)
# answer = return_dict["prediction"] # the text format answer
# masks = return_dict['prediction_masks']  # segmentation masks, list(np.array(n_frames, h, w), ...)


import pickle
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from utils.vggt_utils import DATA_FOLDER
# from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images
# from waypoint_extraction.select_keyframe import get_dataset
# from utils.vggt_utils import extract_vggt_features, sample_keypoints
# from utils.env_utils import (EPISODE_FOLDER, CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST, 
#                              IMAGE_RGB, IMAGE_FORMAT, RLBENCH_TASKS)



# tasks = RLBENCH_TASKS
# get_dataset_func = lambda: get_dataset(
#     tasks,
#     3,
#     None,
#     os.path.join("/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench", "heuristic"),
#     None,
#     DATA_FOLDER,
#     100,
#     None,
#     False,
#     device,
#     num_workers=3,
#     only_train=True,
#     sample_distribution_mode="task_uniform",
# )
# train_dataset = get_dataset_func()
# raw_batch = next(iter(train_dataset))
# batch = {
#     k: v.to(device)
#     for k, v in raw_batch.items()
#     if type(v) == torch.Tensor
# }
# batch["tasks"] = raw_batch["tasks"]
# batch["lang_goal"] = raw_batch["lang_goal"]
# replay_sample = batch


# vggt_model = VGGT().to(device)
# vggt_model.load_state_dict(torch.load("/fs-computility/efm/lvqi/projects/colosseum/SAM2Act/sam2Act_COLOSSEUM/third_libraries/vggt/model.pt"))
# vggt_model.eval()
# rgb_vggt_list = []
# rgb_front_list = []
# rgb_left_list = []
# rgb_right_list = []
# rgb_wrist_list = []
# for j in range(len(replay_sample["episode_idx"])):  # 遍历batch中的每个样本
#     sample = {k: v[j].cpu().numpy() if isinstance(v, torch.Tensor) else v 
#              for k, v in replay_sample.items()}
    
#     index = sample["episode_idx"] # 0 to 99
#     if sample["keypoint_frame"] != -1:
#         i = sample["keypoint_frame"] # or keypoint_frame
#     else:
#         i = sample["sample_frame"] # or keypoint_frame
#     sample_task = replay_sample["tasks"][j]
#     data_path = os.path.join(DATA_FOLDER, f"rlbench/train/{sample_task}/all_variations/episodes")
#     episode_path = os.path.join(data_path, EPISODE_FOLDER % index)

#     # resize for VGGT
#     img_front_path = os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)
#     img_left_path = os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)
#     img_right_path = os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)
#     img_wrist_path = os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)

#     rgb_front = cv2.imread(str(img_front_path))[..., ::-1].copy()
#     # cv2.imwrite(f"debug_runs/images/temp/output{j}_front.jpg", rgb_front) 
#     rgb_front = np.moveaxis((rgb_front / 255.).astype(np.float32), -1, 0)
#     rgb_left = cv2.imread(str(img_left_path))[..., ::-1].copy()
#     # cv2.imwrite(f"debug_runs/images/temp/output{j}_left.jpg", rgb_left) 
#     rgb_left = np.moveaxis((rgb_left / 255.).astype(np.float32), -1, 0)
#     rgb_right = cv2.imread(str(img_right_path))[..., ::-1].copy()
#     # cv2.imwrite(f"debug_runs/images/temp/output{j}_right.jpg", rgb_right) 
#     rgb_right = np.moveaxis((rgb_right / 255.).astype(np.float32), -1, 0)
#     rgb_wrist = cv2.imread(str(img_wrist_path))[..., ::-1].copy()
#     # cv2.imwrite(f"debug_runs/images/temp/output{j}_wrist.jpg", rgb_wrist) 
#     rgb_wrist = np.moveaxis((rgb_wrist / 255.).astype(np.float32), -1, 0)

#     rgb_vggt = load_and_preprocess_images([str(img_front_path), str(img_left_path), str(img_right_path), str(img_wrist_path)])
#     rgb_vggt_list.append(rgb_vggt)
#     rgb_front_list.append(rgb_front)
#     rgb_left_list.append(rgb_left)
#     rgb_right_list.append(rgb_right)
#     rgb_wrist_list.append(rgb_wrist)

# rgb_vggt = torch.stack(rgb_vggt_list)  # (B, V, C, H, W)
# vggt_features = extract_vggt_features(rgb_vggt.to(device), vggt_model, device=device)
# # for key, value in vggt_features.items():
# #     if isinstance(value, torch.Tensor):
# #         print(f"{key}: {value.shape}")
# #     else:
# #         print(f"{key}: {type(value)} (not a tensor)")

# kp_1, kp_2, kp_3, kp_4, valid_kp, mask_1, mask_2, mask_3, mask_4 = sample_keypoints(vggt_features, vggt_model, device=device, 
#                                                                                     num_keypoints=300, min_distance=5)
# # print("kp shape: ", kp_1[0].shape, kp_2[2].shape, kp_3[1].shape, kp_4[0].shape)
# # print("valid_kp shape: ", valid_kp.shape)
# # print("mask shape: ", mask_1.shape, mask_2.shape, mask_3.shape, mask_4.shape)


# mh, mw = vggt_features['image_shape']
# # (B, C=3, H, W)
# rgb_front_resized = F.interpolate(torch.from_numpy(np.array(rgb_front_list)).float().to(device), size=(mh, mw))#, mode='bicubic', align_corners=False
# rgb_left_resized = F.interpolate(torch.from_numpy(np.array(rgb_left_list)).float().to(device), size=(mh, mw))
# rgb_right_resized = F.interpolate(torch.from_numpy(np.array(rgb_right_list)).float().to(device), size=(mh, mw))
# rgb_wrist_resized = F.interpolate(torch.from_numpy(np.array(rgb_wrist_list)).float().to(device), size=(mh, mw))
# # print("resized rgb shape: ", rgb_front_resized.shape, rgb_left_resized.shape, rgb_right_resized.shape, rgb_wrist_resized.shape)

# def sigmoid(tensor, temp=1.0):
#     """ temperature controlled sigmoid

#     takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
#     """
#     exponent = -tensor / temp
#     # clamp the input tensor for stability
#     exponent = torch.clamp(exponent, min=-50, max=50)
#     y = 1.0 / (1.0 + torch.exp(exponent))
#     return y
# from temp_distill import interpolate_features
# import torchvision.transforms as T
# def get_feature(rgbs, pts, model, normalize=True, global_feature=False):
#     target_res = 224 # Or derive from data_config if needed
#     downsample_factor = 8 # Use backbone's downsample factor   
#     patch_size = 14 # model.patch_embed.patch_size[0]
#     refine_conv = torch.nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1)
#     input_transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

#     tgt_size = (int(rgbs.shape[-2] * target_res / rgbs.shape[-1]), target_res)
#     if rgbs.shape[-2] > rgbs.shape[-1]:
#         tgt_size = (target_res, int(rgbs.shape[-1] * target_res / rgbs.shape[-2]))
    
#     patch_h, patch_w = target_res[0] // patch_size, target_res[1] // patch_size
#     rgb_resized = F.interpolate(rgbs, (patch_h * patch_size, patch_w * patch_size))
    
#     resize_factor = [(patch_w * patch_size) / rgbs.shape[-1], (patch_h * patch_size) / rgbs.shape[-2]]
    
#     pts = pts * torch.tensor(resize_factor).to(pts.device)
    
#     if global_feature:
#         result = model.forward_features(input_transform(rgb_resized))
#         global_feat, result = result[:, 0], result[:, 1:]
#     else:    
#         result = model.forward_features(input_transform(rgb_resized)) 
    
#     feature = result.reshape(rgb_resized.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2)

#     feature = refine_conv(feature)
        
#     feature = interpolate_features(feature, pts, h=patch_h * patch_size, w=patch_w * patch_size, patch_size=patch_size, stride=patch_size, normalize=False).permute(0, 2, 1)
#     if normalize:
#         feature = F.normalize(feature, p=2, dim=-1)
    
#     if global_feature:
#         return feature, global_feat

#     return feature
# def calculate_matching_loss(rgb_1_resized, rgb_2_resized, rgb_3_resized, rgb_4_resized, point_map_view_1, point_map_view_2,
#                             point_map_view_3, point_map_view_4, kp_1, kp_2, kp_3, kp_4, thres3d_neg = 0.1):
#     desc_1 = get_feature(rgb_1_resized, kp_1, model=vggt_model, normalize=True)  # (B, N, C)
#     desc_2 = get_feature(rgb_2_resized, kp_2, model=vggt_model, normalize=True)  # (B, N, C)
#     desc_3 = get_feature(rgb_3_resized, kp_3, model=vggt_model, normalize=True)  # (B, N, C)
#     desc_4 = get_feature(rgb_4_resized, kp_4, model=vggt_model, normalize=True)  # (B, N, C)
    
#     pts3d_1 = point_map_view_1[kp_1[...,1].long(), kp_1[...,0].long()]  # (B, N, 3)
#     pts3d_2 = point_map_view_2[kp_2[...,1].long(), kp_2[...,0].long()]  # (B, N, 3)
#     pts3d_3 = point_map_view_3[kp_3[...,1].long(), kp_3[...,0].long()]  # (B, N, 3)
#     pts3d_4 = point_map_view_4[kp_4[...,1].long(), kp_4[...,0].long()]  # (B, N, 3)
    
#     pos_idxs = torch.stack([
#         torch.zeros(desc_1.size(1), dtype=torch.long, device=device),  # [0, 0, ..., 0]
#         torch.arange(desc_1.size(1), device=device),                   # [0, 1, ..., N-1]
#         torch.arange(desc_2.size(1), device=device)                    # [0, 1, ..., N-1]
#     ], dim=1)  # (N, 3) each row [0, i, i]
    
#     eye_mask = torch.eye(desc_1.size(1), device=device).bool().unsqueeze(0)    # (1, N, N) only diag elements are 'True'
#     neg_mask = (torch.cdist(pts3d_1, pts3d_2) > thres3d_neg) & ~eye_mask       # (B, N, N)
    
#     sim = torch.bmm(desc_1, desc_2.transpose(-1, -2))  # (B, N, N) similarities between features of kp_1[i] and kp_2[j]
    
#     pos_sim = sim[pos_idxs[:,0], pos_idxs[:,1], pos_idxs[:,2]]      # (N) similarities of diag elements (positive pairs)
#     rpos = sigmoid(1. - pos_sim, temp=0.01) + 1  # (N)
#     rall = rpos + torch.sum(
#         sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - 1., temp=0.01)  # the first N rows of sim (the whole sim)
#         * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),           # the first N rows of neg_mask (the whole neg_mask)
#         dim=-1
#     )
#     ap1 = rpos / rall
    
#     rpos = sigmoid(1. - pos_sim, temp=0.01) + 1
#     rall = rpos + torch.sum(
#         sigmoid(sim[pos_idxs[:,0], pos_idxs[:,1]] - pos_sim[:, None], temp=0.01) # pos_sim[:, None] (N, 1)
#         * neg_mask[pos_idxs[:,0], pos_idxs[:,1]].float(),
#         dim=-1
#     )
#     ap2 = rpos / rall
    
#     ap = (ap1 + ap2) / 2
#     ap_loss = torch.mean(1. - ap)
    
#     return ap_loss

# ap_loss = calculate_matching_loss(rgb_front_resized, rgb_left_resized, rgb_right_resized, rgb_wrist_resized,
#                                   vggt_features['point_map_view_1'], vggt_features['point_map_view_2'],
#                                   vggt_features['point_map_view_3'], vggt_features['point_map_view_4'],
#                                   kp_1, kp_2, kp_3, kp_4, thres3d_neg = 0.1,) # --- Thresholds (can be tuned) ---

def _create_uniform_pixel_coords_image(resolution: np.ndarray):
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    uniform_pixel_coords = np.concatenate(
        (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
    return uniform_pixel_coords


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(trans, coords)
    transformed_coords_vector = np.transpose(
        transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector,
                    (h, w, -1))


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate(
        [pixel_coords, np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    world_coords_homo = np.concatenate(
        [world_coords, np.ones((h, w, 1))], axis=-1)
    return world_coords_homo                                 


@staticmethod
def pointcloud_from_depth_and_camera(
        depth: np.ndarray, extrinsics: np.ndarray,
        intrinsics: np.ndarray) -> np.ndarray:
    """Converts depth (in meters) to point cloud in word frame.
            depth: A numpy array of size (width, height) in_meters=True
            extrinsics: (4, 4)
            intrinsics: (3, 3)
    :return: A numpy array of size (width, height, 3)
    """
    upc = _create_uniform_pixel_coords_image(depth.shape)
    pc = upc * np.expand_dims(depth, -1)
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    cam_proj_mat_homo = np.concatenate(
        [cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(_pixel_to_world_coords(
        pc, cam_proj_mat_inv), 0)
    world_coords = world_coords_homo[..., :-1][0]
    return world_coords



def visualize_global_coords(global_kps, images, kps, extrinsics, intrinsics):
    """
    全局坐标可视化（解决所有数据类型和维度问题）
    Args:
        global_kps: (B, N, 3)或(N, 3) torch张量或numpy数组
        images: List of (H,W,3) numpy数组
        kps: List of (B, N, 2)或(N, 2) torch张量或numpy数组
        extrinsics: List of (B, 4, 4)或(4, 4) torch张量或numpy数组
        intrinsics: List of (3, 3) numpy数组
    """
    plt.figure(figsize=(20, 15))
    
    # 数据预处理
    global_kps = global_kps[0].cpu().numpy() if torch.is_tensor(global_kps) else global_kps[0]
    extrinsics = [e[0].cpu().numpy() if torch.is_tensor(e) else e[0] for e in extrinsics]
    intrinsics = [i.cpu().numpy() if torch.is_tensor(i) else i for i in intrinsics]
    
    for i in range(4):
        K = intrinsics[i]  # (3,3)
        E = extrinsics[i]  # (4,4)
        R = E[:3, :3]      # 世界→相机的旋转
        t = E[:3, 3]       # 世界→相机的平移
        
        # 核心修正：世界→相机→图像
        cam_coords = (global_kps - t) @ R  # (N,3)
        uv_hom = cam_coords @ K.T          # (N,3)
        uv = uv_hom[:, :2] / uv_hom[:, 2:] # (N,2)
        
        # 可视化
        plt.subplot(2, 2, i+1)
        img = images[i].copy()
        
        # 原始关键点（红色）
        for x, y in kps[i]:
            cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        # 重投影点（绿色）
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), -1)
        
        plt.imshow(img)
        plt.title(f'View {i+1}\nRed:GT, Green:Reprojected')
        plt.axis('off')
        
        # 调试信息
        print(f"View {i+1} - 重投影范围: u[{uv[:,0].min():.1f}, {uv[:,0].max():.1f}], v[{uv[:,1].min():.1f}, {uv[:,1].max():.1f}]")
    
    plt.tight_layout()
    plt.savefig("debug_runs/reprojection_fixed.jpg", dpi=300)
    plt.close()


def convert_to_global_coords(point_maps, kps, extrinsics_global):
    """
    将各视角关键点转换到全局坐标系
    Args:
        point_maps: List of (B, H, W, 3) tensors, 四个视角的3D点图
        kps: List of (B, N, 2) tensors, 四个视角的关键点2D坐标 (x,y)
        extrinsics_global: List of (B, 4, 4) tensors, 四个相机的全局外参
    Returns:
        global_kps: (B, N, 3) tensor, 所有关键点在全局坐标系下的3D坐标
    """
    scale_factor = torch.tensor([128/518, 128/518], device=point_maps[0].device)
    kps = [(kp.float() * scale_factor).long().clamp(0, 127) for kp in kps]
    B, N = kps[0].shape[:2]
    
    # 存储各视角的相机坐标系点
    camera_pts = []
    for i in range(4):
        pts_test = point_maps[i][0, kps[i][0, :, 1], kps[i][0, :, 0]].cpu().numpy()
        print(f"View {i+1} 点图Z值范围:", pts_test[:,2].min(), pts_test[:,2].max())
        # 提取点图坐标（相机坐标系）
        pts = torch.stack([
            point_maps[i][b, kps[i][b, :, 1], kps[i][b, :, 0]]
            for b in range(B)
        ])  # (B,N,3)
        camera_pts.append(pts)
    
    # 三角化（世界坐标系）
    global_kps = triangulate_multi_view(camera_pts, extrinsics_global)
    return global_kps


def triangulate_multi_view(camera_pts, extrinsics):
    """多视角三角化（显式处理外参）"""
    B, N, _ = camera_pts[0].shape
    global_kps = torch.zeros(B, N, 3, device=camera_pts[0].device)
    
    for b in range(B):
        for n in range(N):
            # 构建线性方程组 AX=0
            A = []
            for i in range(4):
                R = extrinsics[i][b, :3, :3]  # 世界→相机的旋转
                t = extrinsics[i][b, :3, 3]   # 世界→相机的平移
                # 公式：相机坐标系点 → 世界坐标系射线
                ray_dir = R.T @ camera_pts[i][b, n]  # 注意转置
                A.append(ray_dir - R.T @ t)
            A = torch.stack(A)  # (4,3)
            _, _, V = torch.svd(A)
            global_kps[b, n] = V[-1, :3] / V[-1, 3]
    return global_kps


def draw_kp(img, kps):
    for x, y in kps:
        cv2.circle(img, (int(x), int(y)), radius=1, color=(255, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return img


def get_3d_keypoints_from_pred(data_test):
    """使用模型预测的相机参数计算关键点的3D坐标"""
    kp_3d_pred = {}
    original_img_size = 518
    [x_min, y_min, z_min, x_max, y_max, z_max] = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6,]
    for i in range(1, 5):
        # 1. 数据加载与检查
        kp_2d = data_test[f'kp_{i}'].squeeze(0).astype(np.float64)
        depth = data_test['vggt_features'][f'depth_pred_{i}'].squeeze(0)
        intrinsics = data_test['vggt_features'][f'intrinsic_{i}'].squeeze(0).astype(np.float64)
        extrinsics = data_test['vggt_features'][f'extrinsic_{i}'].squeeze(0).astype(np.float64)
        print(f"\nView {i} 原始内参:\n{intrinsics}")
        print(f"深度范围: {depth.min():.2f}→{depth.max():.2f}")
        # 2. 计算缩放因子（518→128）
        height, width = depth.shape
        scale = width / original_img_size
        # 3. 关键点坐标转换（保持中心点）
        kp_pixel = np.round(kp_2d * np.array([scale, scale])).astype(int)
        kp_pixel = np.clip(kp_pixel, [0, 0], [width-1, height-1])
        # 4. 内参矩阵处理 
        fx = intrinsics[0,0] * scale
        fy = intrinsics[1,1] * scale
        cx = intrinsics[0,2] #* scale_x
        cy = intrinsics[1,2] #* scale_y
        # 5. 反投影（添加深度缩放）
        kp_3d_cam = np.zeros((len(kp_2d), 3))
        for j, (u, v) in enumerate(kp_pixel):
            z = depth[v,u] 
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            kp_3d_cam[j] = [x, y, z]
        # 6. 世界坐标系转换（添加坐标系检查）
        R, t = extrinsics[:3,:3], extrinsics[:3,3]
        kp_3d_world = (R @ kp_3d_cam.T + t[:, None]).T
        # 边界约束
        x_prediction, y_prediction, z_prediction = kp_3d_world[:, 0], kp_3d_world[:, 1], kp_3d_world[:, 2]
        kp_3d_world[:, 0] = x_min + (kp_3d_world[:, 0] - x_prediction.min()) / (x_prediction.max() - x_prediction.min()) * (x_max - x_min)  # X轴
        kp_3d_world[:, 1] = y_min + (kp_3d_world[:, 1] - y_prediction.min()) / (y_prediction.max() - y_prediction.min()) * (y_max - y_min)  # Y轴
        kp_3d_world[:, 2] = z_min + (kp_3d_world[:, 2] - z_prediction.min()) / (z_prediction.max() - z_prediction.min()) * (z_max - z_min)  # Z轴
        kp_3d_pred[f'kp_{i}'] = kp_3d_world
        print("view ", i, np.unique(kp_3d_world).shape)
    return kp_3d_pred



# def get_3d_keypoints_from_gt(data_test, view_names=['front', 'left_shoulder', 'right_shoulder', 'wrist']):
#     kp_3d_gt = {}
#     scale = 128 / 518  # kp输入是 518x518 图像输出为 128x128
#     for idx, view in enumerate(view_names, 1):
#         kp_2d = data_test[f'kp_{idx}'].squeeze(0).astype(np.float32)
#         kp_pixel = (kp_2d * scale).astype(int)
#         kp_pixel = np.clip(kp_pixel, [0, 0], [127, 127])    # 128x128 图像

#         point_cloud = data_test[f'{view}_point_cloud']      # shape: (3, W, H)
#         _, W, H = point_cloud.shape  # 注意这里是W,H不是H,W
#         print(f"\nView {view} point cloud:")
#         print(f"Shape: {point_cloud.shape}")
#         print(f"X range: [{np.min(point_cloud[...,0]):.2f}, {np.max(point_cloud[...,0]):.2f}]")
#         print(f"Y range: [{np.min(point_cloud[...,1]):.2f}, {np.max(point_cloud[...,1]):.2f}]")
#         print(f"Z range: [{np.min(point_cloud[...,2]):.2f}, {np.max(point_cloud[...,2]):.2f}]")

#         kp_3d = []
#         valid_count = 0
#         for x, y in kp_pixel:
#             if 0 <= x < W and 0 <= y < H:
#                 # 直接获取世界坐标系下的点（无需再转换）
#                 xyz_world = point_cloud[:, x, y]
#                 if not np.all(xyz_world == 0):  # 过滤零值点
#                     kp_3d.append(xyz_world)
#                     valid_count += 1
#                 else:
#                     kp_3d.append([0.0, 0.0, 0.0])  # 保证形状一致
#             else:
#                 kp_3d.append([0.0, 0.0, 0.0])  # 保证形状一致
#         kp_3d_gt[f'kp_{idx}'] = np.array(kp_3d, dtype=np.float32)
#         print(f"Extracted {valid_count} valid keypoints (from {len(kp_pixel)} raw points)")
#     return kp_3d_gt 





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





def visualize_pred_keypoints(data_test, kp_3d_pred):
    """仅可视化预测的关键点（因为坐标系可能不对齐）"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for i in range(1, 5):
        kp = kp_3d_pred[f'kp_{i}']
        ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], c=colors[i-1], s=50, marker='o', label=f'KP_{i} Pred')   
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Predicted 3D Keypoints (in Model\'s Coordinate Frame)')
    plt.savefig('debug_runs/visualize_pred_keypoints.png', dpi=300, bbox_inches='tight')
    plt.close()


# def visualize_comparison(data_test, kp_3d_pred, save_dir="debug_runs/comparison_results"):
#     """
#     多模态比较预测关键点与原始点云
#     包含：3D点云对比、2D投影对比、深度图叠加、误差分析
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # ================= 1. 3D点云对比可视化 =================
#     fig = plt.figure(figsize=(18, 10))
#     # 3D视图1
#     ax1 = fig.add_subplot(121, projection='3d')
#     plot_3d_comparison(ax1, data_test, kp_3d_pred, view='overview')
#     # 3D视图2（不同视角）
#     ax2 = fig.add_subplot(122, projection='3d')
#     plot_3d_comparison(ax2, data_test, kp_3d_pred, view='top_down')
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/3d_comparison.png", dpi=300)
#     plt.close()
#     # ================= 2. 各相机视角2D投影对比 =================
#     view_mapping = {
#         'front': 'front',
#         'left_shoulder': 'left_shoulder',
#         'right_shoulder': 'right_shoulder',
#         'wrist': 'wrist'
#     }
#     original_keypoints = {
#         'front': data_test['kp_1'].squeeze(0) * 128 / 518,
#         'left_shoulder': data_test['kp_2'].squeeze(0) * 128 / 518,
#         'right_shoulder': data_test['kp_3'].squeeze(0) * 128 / 518,
#         'wrist': data_test['kp_4'].squeeze(0) * 128 / 518
#     }    
#     for view_name in view_mapping.keys():
#         fig = plt.figure(figsize=(15, 6))
#         # RGB图像
#         ax1 = fig.add_subplot(131)
#         plot_2d_projection(ax1, data_test, kp_3d_pred, view_name, overlay_type='rgb', original_kps=original_keypoints[view_name])
#         ax1.set_title(f'{view_name} RGB Projection')
#         # 深度图
#         ax2 = fig.add_subplot(132)
#         plot_2d_projection(ax2, data_test, kp_3d_pred, view_name, overlay_type='depth')
#         ax2.set_title(f'{view_name} Depth Projection')
#         # 误差热力图
#         ax3 = fig.add_subplot(133)
#         plot_error_heatmap(ax3, data_test, kp_3d_pred, view_name)
#         ax3.set_title(f'{view_name} Error Heatmap')
#         plt.tight_layout()
#         plt.savefig(f"{save_dir}/2d_{view_name}_comparison.png", dpi=300)
#         plt.close()





def plot_3d_comparison(ax, data_test, kp_3d_pred, view='overview'):
    """3D点云可视化"""
    # 获取ground truth点云（各视角先reshape为(N,3)再合并）
    gt_points = []
    for view_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist']:
        pc = data_test[f'{view_name}_point_cloud'].transpose(1,2,0)  # shape: (W,H,3)
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
    for i in range(1, 5):
        kp = kp_3d_pred[f'kp_{i}']  # (300, 3)
        valid_kp = kp[~np.all(kp == 0, axis=1)]  # 过滤全零点
        if len(valid_kp) > 0:
            ax.scatter(valid_kp[:,0], valid_kp[:,1], valid_kp[:,2],
                      c=colors[i-1], s=30, marker='o',
                      label=f'KP_{i} ({len(valid_kp)} pts)')
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
    # 获取当前视角参数（添加分辨率处理）
    rgb = data_test[f'{view_name}_rgb'].transpose(1,2,0)
    depth = data_test[f'{view_name}_depth'].squeeze(0)
    intrinsics = data_test[f'{view_name}_camera_intrinsics']
    extrinsics = data_test[f'{view_name}_camera_extrinsics'] 
    H, W = depth.shape

    # 显示背景（添加分辨率适配）
    if overlay_type == 'rgb':
        img = rgb.copy()
        if original_kps is not None:
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
        # 世界→相机坐标系 (使用外参的逆变换)
        R = extrinsics[:3,:3]
        t = extrinsics[:3,3]
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
    # 获取原始关键点和预测投影
    original_kps = data_test[f'kp_{["front", "left_shoulder", "right_shoulder", "wrist"].index(view_name)+1}'].squeeze(0)
    scale = 128 / 518
    # 收集误差数据
    points = []
    errors = []
    for kp_id in range(1, 5):
        if kp_id-1 == ['front', 'left_shoulder', 'right_shoulder', 'wrist'].index(view_name):
            kp_world = kp_3d_pred[f'kp_{kp_id}']
            extrinsics = data_test[f'{view_name}_camera_extrinsics']
            intrinsics = data_test[f'{view_name}_camera_intrinsics']
            R = extrinsics[:3,:3]
            t = extrinsics[:3,3]
            kp_cam = (kp_world - t) @ R
            kp_pixel = kp_cam @ intrinsics.T
            kp_pixel = kp_pixel[:,:2] / kp_pixel[:,2:]
            if len(kp_pixel) > 0:
                error = np.linalg.norm(kp_pixel - original_kps * scale, axis=1)
                points.extend(kp_pixel.tolist())
                errors.extend(error.tolist())
    # 绘制误差图
    if points:
        points = np.array(points)
        errors = np.array(errors)
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
        
def extract_single_sample(batch_data, pred_kp_dict, index=0):
    """从批量数据中提取单个样本"""
    single_sample = {}
    # 处理字典中的张量字段
    for k, v in batch_data.items():
        if isinstance(v, torch.Tensor):
            single_sample[k] = v[index].unsqueeze(0)  # 保持batch维度
        else:
            single_sample[k] = v
    
    # 处理关键点预测
    single_kp = {}
    for k, v in pred_kp_dict.items():
        single_kp[k] = v[index].unsqueeze(0)  # 保持batch维度
    
    return single_sample, single_kp


if __name__ == "__main__":
    device = "cuda:1"
    task = ["slide_block_to_color_target"]
    number = 946
    database = "/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/rlbench/heuristic"

    from third_libraries.vggt.avt_vggt.utils.vggt_utils import visualize_comparison, visualize_pc_transformation
    from sam2act_colosseum.train_rvt import (DATA_FOLDER, IMAGE_SIZE, SCENE_BOUNDS, CAMERAS, 
                                             get_dataset, rvt_agent, MVT, mvt_cfg_mod, get_num_feat, get_tasks, exp_cfg_mod)
    from sam2act_colosseum.rvt.rvt_agent import (_preprocess_inputs, rvt_utils, mvt_utils, apply_se3_aug_con,
                                                 aug_utils, autocast, get_3d_keypoints_from_gt)
    
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    exp_cfg.merge_from_file("/fs-computility/efm/lvqi/projects/colosseum/SAM2Act/sam2Act_COLOSSEUM/sam2act_colosseum/configs/rvt2.yaml")
    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= 1 * exp_cfg.bs
# def run_cpu_pipeline():
#     # 加载数据 (确保数据在CPU上)
#     raw_batch = next(iter(train_dataset))
#     batch = {
#         **{k: v.cpu() for k,v in raw_batch.items() if isinstance(v, torch.Tensor)},
#         "tasks": raw_batch["tasks"],
#         "lang_goal": raw_batch["lang_goal"]
#     }

#     # CPU预处理
#     obs, pcd = cpu_preprocess_inputs(batch, CAMERAS)
#     kp_3d_pred = cpu_get_3d_keypoints(batch)
    
#     # 模拟网络计算
#     net_sim = CPUNetworkSimulator()
#     with torch.no_grad():
#         # 模拟MVT1
#         feature_1, feature_2, _, correspondence = net_sim.get_matching_feature(
#             net_sim.render(pc=[p[1] for p in obs], img_feat=[p[0] for p in obs]),
#             {'normalize': True}
#         )
        
#         # 计算模拟损失
#         sim_mat = torch.rand(2, batch_size, 100, 100)  # 模拟相似度矩阵
#         dist_mask = torch.ones_like(sim_mat, dtype=bool)
#         ap_loss = torch.mean(1.0 - (torch.sigmoid((1-sim_mat)/0.01) / 
#                                  (torch.sigmoid((1-sim_mat)/0.01) + 
#                                   torch.sum(torch.sigmoid((sim_mat-1)/0.01)*dist_mask.float(), dim=(2,3)))))
    
#     print(f"CPU模拟完成！AP Loss: {ap_loss.item():.4f}")
    exp_cfg.freeze()
    BATCH_SIZE_TRAIN = 5
    NUM_TRAIN = 10
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * 1))
    EPOCHS = exp_cfg.epochs
    tasks, base_replay_dir = get_tasks(exp_cfg)
    TRAIN_REPLAY_STORAGE_DIR = os.path.join(base_replay_dir, "heuristic")
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        False,
        device,
        num_workers=3,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
    )
    train_dataset = get_dataset_func()
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    mvt_cfg.merge_from_file("/fs-computility/efm/lvqi/projects/colosseum/SAM2Act/sam2Act_COLOSSEUM/sam2act_colosseum/mvt/configs/rvt2.yaml")
    
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
    mvt_cfg.freeze()
    rvt = MVT(
            renderer_device=device,
            **mvt_cfg,
        ).to(device)
    agent = rvt_agent.RVTAgent(
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
    agent.build(training=True, device=device)
    agent.train()
    data_iter = iter(train_dataset)
    raw_batch = next(data_iter)
    # ===== 仅加载相关字段 =====
    del train_dataset, data_iter
    required_fields = {
        "gripper_pose",
        *[f"{cam}_rgb" for cam in CAMERAS],
        *[f"{cam}_depth" for cam in CAMERAS],
        *[f"{cam}_point_cloud" for cam in CAMERAS],
        *[f"{cam}_camera_intrinsics" for cam in CAMERAS],
        *[f"{cam}_camera_extrinsics" for cam in CAMERAS],
        *[f"kp_{idx}" for idx in range(1, len(CAMERAS)+1)]
    }
    # ==============================

    batch = {
        k: v.to(device) 
        for k, v in raw_batch.items()
        if type(v) == torch.Tensor and k in required_fields
    }
    replay_sample = batch


    action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
    obs, pcd = _preprocess_inputs(replay_sample, CAMERAS)

    kp_3d_pred = get_3d_keypoints_from_gt(replay_sample)    # dict_keys(['kp_1', 'kp_2', 'kp_3', 'kp_4']) [b, 300, 3]
    # single_sample, single_kp = extract_single_sample(replay_sample, kp_3d_pred, index=0)
    # visualize_comparison(single_sample, single_kp)

    match_key_pts = []
    for i in range(1,5):
        kp_3d = kp_3d_pred[f'kp_{i}']  # (B,N,3)
        match_key_pts.append(kp_3d)
    with torch.no_grad():
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        batch_size = match_key_pts[0].shape[0]  
        match_key_pts_tensor = torch.stack(match_key_pts)  # [V, B, 300, 3]
        match_pc = match_key_pts_tensor.permute(1, 0, 2, 3).reshape(batch_size, -1, 3)  # [B,300*V,3]

        if True:
            action_trans_con, action_rot, pc = apply_se3_aug_con(
                pcd=pc,
                action_gripper_pose=action_gripper_pose,
                bounds=torch.tensor(SCENE_BOUNDS),
                trans_aug_range=torch.tensor([0.125, 0.125, 0.125]),
                rot_aug_range=torch.tensor([0.0, 0.0, 45.0]),
            )
            action_trans_con = torch.tensor(action_trans_con).to(pc.device)
            action_rot = torch.tensor(action_rot).to(pc.device)

        # TODO: vectorize
        action_rot = action_rot.cpu().numpy()
        for i, _action_rot in enumerate(action_rot):
            _action_rot = aug_utils.normalize_quaternion(_action_rot)
            if _action_rot[-1] < 0:
                _action_rot = -_action_rot
            action_rot[i] = _action_rot
        
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, SCENE_BOUNDS, no_op=False
        )
        wpt = [x[:3] for x in action_trans_con]
        match_pc, _ = rvt_utils.move_pc_in_bound(match_pc, match_pc, SCENE_BOUNDS, no_op=False)
        # len = B (750~964, 3)
        
        wpt_local = []
        rev_trans = []
        for _pc, _wpt in zip(pc, wpt):
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                _wpt,
                with_mean_or_bounds=False,
                scene_bounds=SCENE_BOUNDS,
            )
            wpt_local.append(a.unsqueeze(0))
            rev_trans.append(b)

        wpt_local = torch.cat(wpt_local, axis=0)
        # [ 0.4512,  0.2907, -0.5335], [ 0.1532,  0.0524, -0.0217], [ 0.6513, -0.6185, -0.4542], [ 0.2186,  0.0035,  0.6391], 
        # [ 0.4095, -0.1798, -0.5466], [-0.1481, -0.2013,  0.1838], [ 0.2139,  0.2128, -0.5897], [ 0.2235, -0.3586, -0.4839],
        # [ 0.1223, -0.2483, -0.0022], [ 0.2090, -0.1601, -0.5551], [ 0.3865, -0.4674,  0.0398], [-0.1015, -0.1688, -0.2051],
        # [ 0.4919,  0.1180,  0.2676], [ 0.4730, -0.3728,  0.2655], [-0.3374,  0.5906, -0.7290], [ 0.2854, -0.5580, -0.0826]

        # TODO: Vectorize
        pc = [
            mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=False,
                scene_bounds=SCENE_BOUNDS,
            )[0]
            for _pc in pc
        ]
        match_pc = [mvt_utils.place_pc_in_cube(_pc, with_mean_or_bounds=False,
                                                scene_bounds=SCENE_BOUNDS,)[0] 
                                                for _pc in match_pc] # -1~1



        bs = len(pc)
        nc = 3
        h = w = agent._net_mod.img_size

        img_aug = 0

        dyn_cam_info = None

    with autocast(enabled=True):
        match_input_dict = {
            'normalize': True,
            'match_pc': match_pc,
        }
        with torch.no_grad():
            if (agent._network.img_aug_2 != 0):
                for x in img_feat:
                    stdv = agent._network.img_aug_2 * torch.rand(1, device=x.device)
                    # values in [-stdv, stdv]
                    noise = stdv * ((2 * torch.rand(*x.shape, device=x.device)) - 1)
                    x = x + noise
            img = agent._network.render(
                pc=pc,
                img_feat=img_feat,
                img_aug=img_aug,
                mvt1_or_mvt2=True,
                dyn_cam_info=None,
            )

        wpt_local_stage_one = wpt_local
        wpt_local_stage_one = wpt_local_stage_one.clone().detach()

        if True: # agent._network.mvt1.no_feat
            feature_1, feature_2, feature_3, correspondence = agent._network.mvt1.get_matching_feature(img.clone(), match_input_dict)
            out = {"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3, "match_correspondence": correspondence}
        print("mvt1 finished.")
        pc_before = pc
        if agent._network.stage_two:
            with torch.no_grad():
                wpt_local_stage_one_noisy = mvt_utils.add_uni_noi(
                    wpt_local_stage_one.clone().detach(), 2 * agent._network.st_wpt_loc_aug
                )
                pc, rev_trans = mvt_utils.trans_pc(
                    pc, loc=wpt_local_stage_one_noisy, sca=agent._network.st_sca
                )
                match_pc2, _ = mvt_utils.trans_pc(
                    match_input_dict['match_pc'], loc=wpt_local_stage_one, sca=agent._network.st_sca
                )
                visualize_pc_transformation(pc_before, match_input_dict['match_pc'], pc, match_pc2, 
                              title_before="Before", title_after="After", sample_idx=0)
                match_input_dict_st2 = {
                    'normalize': True,
                    'match_pc': match_pc2,
                }
                if agent._network.st_wpt_loc_inp_no_noise:
                    wpt_local2, _ = mvt_utils.trans_pc(
                        wpt_local, loc=wpt_local_stage_one_noisy, sca=agent._network.st_sca
                    )
                else:
                    wpt_local2, _ = mvt_utils.trans_pc(
                        wpt_local, loc=wpt_local_stage_one, sca=agent._network.st_sca
                    )
                    

                img = agent._network.render(
                    pc=pc,
                    img_feat=img_feat,
                    img_aug=img_aug,
                    mvt1_or_mvt2=False,
                    dyn_cam_info=None,
                )

            # if agent._network.mvt2.no_feat:
            feature_1, feature_2, feature_3, correspondence = agent._network.mvt2.get_matching_feature(img.clone(), match_input_dict_st2)
            out_mvt2 = {"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3, "match_correspondence": correspondence}
            print("mvt2 finished.")

            out["wpt_local1"] = wpt_local_stage_one_noisy
            out["rev_trans"] = rev_trans
            out["mvt2"] = out_mvt2


    with autocast(enabled=agent.amp):
        ap_loss = agent.calculate_matching_loss(out, thres3d_neg = 0.1) # Thresholds (can be tuned) 
        print("ap_loss = ", ap_loss)






    # with open(f"{database}/{task}/{number}.replay", "rb") as f:
    #     data_test = pickle.load(f)
    # print(data_test.keys())
    # # 获取原始关键点
    # kp_3d_pred = get_3d_keypoints_from_gt(data_test)

    # # # 查找共有关键点
    # # common_keypoints, stats = get_common_3d_keypoints(kp_3d_pred, 0.05)

    # # print("合并后的点及其来源视角:")
    # # for pt, views in common_keypoints:
    # #     print(f"Point: {pt.round(2)}, Views: {views}")

    # # print("\n统计信息:")
    # # print(f"原始总点数: {stats['total_points']}")
    # # print(f"合并后的点数: {stats['merged_count']}")
    # # print(f"2-view共有点: {stats['view_coverage'][2]}")
    # # print(f"3-view共有点: {stats['view_coverage'][3]}")
    # # print(f"4-view共有点: {stats['view_coverage'][4]}")


    # # padded_kp_3d_common = {f'kp_{i}': np.zeros_like(kp_3d_pred[f'kp_{i}']) for i in range(1, 5)}
    # # for pt, view_indices in common_keypoints:
    # #     for view_idx in view_indices:
    # #         # 找到该视角第一个可用的空位（全零的位置）
    # #         view_key = f'kp_{view_idx}'
    # #         mask = np.all(padded_kp_3d_common[view_key] == 0, axis=1)
    # #         available_idx = np.where(mask)[0]
            
    # #         if len(available_idx) > 0:
    # #             padded_kp_3d_common[view_key][available_idx[0]] = pt

     
    # # visualize_comparison(data_test, padded_kp_3d_common) 
    # # visualize_comparison(data_test, kp_3d_pred)


    

        
    

    # gt_points = []
    # for view_name in ['front', 'left_shoulder', 'right_shoulder', 'wrist']:
    #     pc = data_test[f'{view_name}_point_cloud'].transpose(1,2,0)  # shape: (W,H,3)
    #     gt_points.append(pc.reshape(-1,3))          # (16384, 3)
    #     test_point = pc.reshape(-1,3)
    #     print(test_point.shape, test_point[0], test_point[100], test_point[1000])
    # gt_points = np.concatenate(gt_points, axis=0)   # (65536, 3)
    # # 自动计算合理的坐标范围（保留10%边界）
    # x_min, x_max = np.percentile(gt_points[:,0], [5, 95])
    # y_min, y_max = np.percentile(gt_points[:,1], [5, 95]) 
    # z_min, z_max = np.percentile(gt_points[:,2], [5, 95])
    # bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    # # 过滤无效点（全零点和NaN）
    # valid_mask = ~np.all(gt_points == 0, axis=1) & ~np.isnan(gt_points).any(axis=1)
    # gt_points = gt_points[valid_mask]
    # 绘制ax.scatter(gt_points[:,0], gt_points[:,1], gt_points[:,2],

    # ax.scatter(kp_3d_pred[f'kp_{i}'][:,0], kp_3d_pred[f'kp_{i}'][:,1], kp_3d_pred[f'kp_{i}'][:,2],


    



# ['left_shoulder_rgb' (3, 128, 128), 'left_shoulder_depth' (1, 128, 128), 'left_shoulder_point_cloud' (3, 128, 128), 
#  'front_camera_extrinsics' (4, 4), 'front_camera_intrinsics' (3, 3), 
#  'kp_1' (1, 300, 2), 'vggt_features', 
#  'lang_goal_embs', 'action', 'reward', 'terminal', 'timeout']
# # NEW
# vggt_features['point_map_view_1', 'point_map_view_2', 'point_map_view_3', 'point_map_view_4', 
#               'point_conf_view_1', 'extrinsic_1', 'extrinsic_2', 'extrinsic_3', 'extrinsic_4', 
#               'intrinsic_1', 'intrinsic_2', 'intrinsic_3', 'intrinsic_4', 
#               'depth_pred_1', 'depth_pred_2', 'depth_pred_3', 'depth_pred_4']