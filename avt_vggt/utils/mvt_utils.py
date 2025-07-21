import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation
from pytorch3d import transforms as torch3d_tf
from rlbench.backend.observation import Observation
from .env_utils import Summary, ScalarSummary, HistogramSummary, ImageSummary, ActResult

# Contants
VOXEL_SIZES = [100]                                 # 100x100x100 voxels
LOW_DIM_SIZE = 18                                   # {left_finger_joint, right_finger_joint, gripper_open, timestep}

DEMO_AUGMENTATION_EVERY_N = 10                      # sample n-th frame in demo
ROTATION_RESOLUTION = 5                             # degree increments per axis

# ========================= basic tools ========================= 

# ----- for processing inputs -----

def stack_on_channel(x):
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)

# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

# original tools to process pc and img_feat
def get_pc_img_feat(obs, pcd):
    """
    preprocess both the point cloud and rgb data to shape (b, H * W * 4, 3)
    copied from SAM2Act

    """
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    img_feat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1)
    img_feat = (img_feat + 1) / 2
    return pc, img_feat

def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


# ----- for transforming point cloud from original frame to scene frame -----

def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ], device = pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


# ----- for select_feat_from_hm and select_feat_from_hm_cache -----

# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)

def select_feat_from_hm(
    pt_cam: torch.Tensor, hm: torch.Tensor, pt_cam_wei: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    """
    :param pt_cam:
        continuous location of point coordinates from where value needs to be
        selected. it is of size [nc, npt, 2], locations in pytorch3d screen
        notations
    :param hm: size [nc, nw, h, w]
    :param pt_cam_wei:
        some predifined weight of size [nc, npt], it is used along with the
        distance weights
    :return:
        tuple with the first element being the wighted average for each point
        according to the hm values. the size is [nc, npt, nw]. the second and
        third elements are intermediate values to be used while chaching
    """
    nc, nw, h, w = hm.shape
    npt = pt_cam.shape[1]
    if pt_cam_wei is None:
        pt_cam_wei = torch.ones([nc, npt], device = hm.device)

    # giving points outside the image zero weight
    pt_cam_wei[pt_cam[:, :, 0] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 1] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 0] > (w - 1)] = 0
    pt_cam_wei[pt_cam[:, :, 1] > (h - 1)] = 0

    pt_cam = pt_cam.unsqueeze(2).repeat([1, 1, 4, 1])
    # later used for calculating weight
    pt_cam_con = pt_cam.detach().clone()

    # getting discrete grid location of pts in the camera image space
    pt_cam[:, :, 0, 0] = torch.floor(pt_cam[:, :, 0, 0])
    pt_cam[:, :, 0, 1] = torch.floor(pt_cam[:, :, 0, 1])
    pt_cam[:, :, 1, 0] = torch.floor(pt_cam[:, :, 1, 0])
    pt_cam[:, :, 1, 1] = torch.ceil(pt_cam[:, :, 1, 1])
    pt_cam[:, :, 2, 0] = torch.ceil(pt_cam[:, :, 2, 0])
    pt_cam[:, :, 2, 1] = torch.floor(pt_cam[:, :, 2, 1])
    pt_cam[:, :, 3, 0] = torch.ceil(pt_cam[:, :, 3, 0])
    pt_cam[:, :, 3, 1] = torch.ceil(pt_cam[:, :, 3, 1])
    pt_cam = pt_cam.long()  # [nc, npt, 4, 2]
    # since we are taking modulo, points at the edge, i,e at h or w will be
    # mapped to 0. this will make their distance from the continous location
    # large and hence they won't matter. therefore we don't need an explicit
    # step to remove such points
    pt_cam[:, :, :, 0] = torch.fmod(pt_cam[:, :, :, 0], int(w))
    pt_cam[:, :, :, 1] = torch.fmod(pt_cam[:, :, :, 1], int(h))
    pt_cam[pt_cam < 0] = 0

    # getting normalized weight for each discrete location for pt
    # weight based on distance of point from the discrete location
    # [nc, npt, 4]
    pt_cam_dis = 1 / (torch.sqrt(torch.sum((pt_cam_con - pt_cam) ** 2, dim=-1)) + 1e-10)
    pt_cam_wei = pt_cam_wei.unsqueeze(-1) * pt_cam_dis
    _pt_cam_wei = torch.sum(pt_cam_wei, dim=-1, keepdim=True)
    _pt_cam_wei[_pt_cam_wei == 0.0] = 1
    # cached pt_cam_wei in select_feat_from_hm_cache
    pt_cam_wei = pt_cam_wei / _pt_cam_wei  # [nc, npt, 4]

    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    pt_cam = pt_cam.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
    # cached pt_cam in select_feat_from_hm_cache
    pt_cam = (pt_cam[:, :, 1] * w) + pt_cam[:, :, 0]  # [nc, 4 * npt]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, npt, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val, pt_cam, pt_cam_wei


# ----- for action calculation -----

def rand_dist(size, min=-1.0, max=1.0, device="cuda"):
    # shape = size random tensor in range [min, max]
    return (max - min) * torch.rand(size, device=device) + min

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True) 

def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = Rotation.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler

    return euler

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy


# ----- for loading agent from checkpoint -----

class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, deterministic: bool) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass

    @abstractmethod
    def update_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def act_summaries(self) -> List[Summary]:
        pass

    @abstractmethod
    def load_weights(self, savedir: str) -> None:
        pass

    @abstractmethod
    def save_weights(self, savedir: str) -> None:
        pass

class PreprocessAgent(Agent):

    def __init__(self,
                 pose_agent: Agent,
                 norm_rgb: bool = True):
        self._pose_agent = pose_agent
        self._norm_rgb = norm_rgb

    def build(self, training: bool, device: torch.device = None):
        self._pose_agent.build(training, device)

    def _norm_rgb_(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def update(self, step: int, replay_sample: dict) -> dict:
        # Samples are (B, N, ...) where N is number of buffers/tasks. This is a single task setup, so 0 index.
        replay_sample = {k: v[:, 0] if len(v.shape) > 2 else v for k, v in replay_sample.items()}
        for k, v in replay_sample.items():
            if self._norm_rgb and 'rgb' in k:
                replay_sample[k] = self._norm_rgb_(v)
            else:
                replay_sample[k] = v.float()
        self._replay_sample = replay_sample
        return self._pose_agent.update(step, replay_sample)

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        # observation = {k: torch.tensor(v) for k, v in observation.items()}
        for k, v in observation.items():
            if self._norm_rgb and 'rgb' in k:
                observation[k] = self._norm_rgb_(v)
            else:
                observation[k] = v.float()
        act_res = self._pose_agent.act(step, observation, deterministic)
        act_res.replay_elements.update({'demo': False})
        return act_res

    def update_summaries(self) -> List[Summary]:
        prefix = 'inputs'
        demo_f = self._replay_sample['demo'].float()
        demo_proportion = demo_f.mean()
        tile = lambda x: torch.squeeze(
            torch.cat(x.split(1, dim=1), dim=-1), dim=1)
        sums = [
            ScalarSummary('%s/demo_proportion' % prefix, demo_proportion),
            HistogramSummary('%s/low_dim_state' % prefix,
                    self._replay_sample['low_dim_state']),
            HistogramSummary('%s/low_dim_state_tp1' % prefix,
                    self._replay_sample['low_dim_state_tp1']),
            ScalarSummary('%s/low_dim_state_mean' % prefix,
                    self._replay_sample['low_dim_state'].mean()),
            ScalarSummary('%s/low_dim_state_min' % prefix,
                    self._replay_sample['low_dim_state'].min()),
            ScalarSummary('%s/low_dim_state_max' % prefix,
                    self._replay_sample['low_dim_state'].max()),
            ScalarSummary('%s/timeouts' % prefix,
                    self._replay_sample['timeout'].float().mean()),
        ]

        for k, v in self._replay_sample.items():
            if 'rgb' in k or 'point_cloud' in k:
                if 'rgb' in k:
                    # Convert back to 0 - 1
                    v = (v + 1.0) / 2.0
                sums.append(ImageSummary('%s/%s' % (prefix, k), tile(v)))

        if 'sampling_probabilities' in self._replay_sample:
            sums.extend([
                HistogramSummary('replay/priority',
                                 self._replay_sample['sampling_probabilities']),
            ])
        sums.extend(self._pose_agent.update_summaries())
        return sums

    def act_summaries(self) -> List[Summary]:
        return self._pose_agent.act_summaries()

    def load_weights(self, savedir: str):
        self._pose_agent.load_weights(savedir)

    def save_weights(self, savedir: str):
        self._pose_agent.save_weights(savedir)

    def reset(self) -> None:
        self._pose_agent.reset()

class PreprocessAgent2(PreprocessAgent):
    def eval(self):
        self._pose_agent._qattention_agents[0]._q.eval()

    def train(self):
        self._pose_agent._qattention_agents[0]._q.train()

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self._device = self._pose_agent._qattention_agents[0]._device


# ========================= functions only used in training ========================= 

# ----- for applying SE3 augmentation to point clouds and actions -----

def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation.
    :param pcd:
        Either:
        - list of point clouds [[bs, 3, H, W], ...] for N cameras
        - point cloud [bs, 3, H, W]
        - point cloud [bs, 3, num_point]
        - point cloud [bs, num_point, 3]
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds in the same format as input
    """
    # batch bounds if necessary

    # for easier compatibility
    single_pc = False
    if not isinstance(pcd, list):
        single_pc = True
        pcd = [pcd]

    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        permute_p = False
        if len(p.shape) == 3:
            if p_shape[-1] == 3:
                num_points = p_shape[-2]
                p = p.permute(0, 2, 1)
                permute_p = True
            elif p_shape[-2] == 3:
                num_points = p_shape[-1]
            else:
                assert False, p_shape

        elif len(p.shape) == 4:
            assert p_shape[-1] != 3, p_shape[-1]
            assert p_shape[-2] != 3, p_shape[-2]
            num_points = p_shape[-1] * p_shape[-2]

        else:
            assert False, len(p.shape)

        action_trans_3x1 = (
            action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )
        trans_shift_3x1 = (
            trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1], device = p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(
            p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=1,
        )

        # shift back the origin
        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        )
        if permute_p:
            perturbed_p_flat_3x1 = torch.permute(perturbed_p_flat_3x1, (0, 2, 1))
        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)

    if single_pc:
        perturbed_pcd = perturbed_pcd[0]

    return perturbed_pcd

def apply_se3_aug_con(
    pcd,
    action_gripper_pose,
    bounds,
    trans_aug_range,
    rot_aug_range,
    scale_aug_range=False,
    single_scale=True,
    ver=2,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: [bs, num_points, 3]
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param bounds: metric scene bounds
        Either:
        - [bs, 6]
        - [6]
    :param trans_aug_range: range of translation augmentation
        [x_range, y_range, z_range]; this is expressed as the percentage of the scene bound
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param scale_aug_range: range of scale augmentation [x_range, y_range, z_range]
    :param single_scale: whether we preserve the relative dimensions
    :return: perturbed action_gripper_pose,  pcd
    """

    # batch size
    bs = pcd.shape[0]
    device = pcd.device

    if len(bounds.shape) == 1:
        bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
    if len(trans_aug_range.shape) == 1:
        trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
    if len(rot_aug_range.shape) == 1:
        rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1).to(device)

    # identity matrix
    identity_4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]

    if ver == 1:
        action_gripper_quat_wxyz = torch.cat(
            (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
        )
        action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)

    elif ver == 2:
        # applying gimble fix to calculate a new action_gripper_rot
        r = Rotation.from_quat(action_gripper_pose[:, 3:7].cpu().numpy())
        euler = r.as_euler("xyz", degrees=True)
        euler = sensitive_gimble_fix(euler)
        action_gripper_rot = torch.tensor(
            Rotation.from_euler("xyz", euler, degrees=True).as_matrix(),
            device=action_gripper_pose.device,
        )
    else:
        assert False

    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    # sample translation perturbation with specified range
    # augmentation range is a percentage of the scene bound
    trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
    # rand_dist samples value from -1 to 1
    trans_shift = trans_range * rand_dist((bs, 3), device=device)

    # apply bounded translations
    bounds_x_min, bounds_x_max = bounds[:, 0], bounds[:, 3]
    bounds_y_min, bounds_y_max = bounds[:, 1], bounds[:, 4]
    bounds_z_min, bounds_z_max = bounds[:, 2], bounds[:, 5]

    trans_shift[:, 0] = torch.clamp(
        trans_shift[:, 0],
        min=bounds_x_min - action_gripper_trans[:, 0],
        max=bounds_x_max - action_gripper_trans[:, 0],
    )
    trans_shift[:, 1] = torch.clamp(
        trans_shift[:, 1],
        min=bounds_y_min - action_gripper_trans[:, 1],
        max=bounds_y_max - action_gripper_trans[:, 1],
    )
    trans_shift[:, 2] = torch.clamp(
        trans_shift[:, 2],
        min=bounds_z_min - action_gripper_trans[:, 2],
        max=bounds_z_max - action_gripper_trans[:, 2],
    )

    trans_shift_4x4 = identity_4x4.detach().clone()
    trans_shift_4x4[:, 0:3, 3] = trans_shift

    roll = torch.deg2rad(rot_aug_range[:, 0:1] * rand_dist((bs, 1), device=device))
    pitch = torch.deg2rad(rot_aug_range[:, 1:2] * rand_dist((bs, 1), device=device))
    yaw = torch.deg2rad(rot_aug_range[:, 2:3] * rand_dist((bs, 1), device=device))
    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
        torch.cat((roll, pitch, yaw), dim=1), "XYZ"
    )
    rot_shift_4x4 = identity_4x4.detach().clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    if ver == 1:
        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
    elif ver == 2:
        perturbed_action_gripper_4x4 = identity_4x4.detach().clone()
        perturbed_action_gripper_4x4[:, 0:3, 3] = action_gripper_4x4[:, 0:3, 3]
        perturbed_action_gripper_4x4[:, :3, :3] = torch.bmm(
            rot_shift_4x4.transpose(1, 2)[:, :3, :3], action_gripper_4x4[:, :3, :3]
        )
    else:
        assert False

    perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

    # convert transformation matrix to translation + quaternion
    perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
    perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
        perturbed_action_gripper_4x4[:, :3, :3]
    )
    perturbed_action_quat_xyzw = (
        torch.cat(
            [
                perturbed_action_quat_wxyz[:, 1:],
                perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )

    # TODO: add scale augmentation

    # apply perturbation to pointclouds
    # takes care for not moving the point out of the image
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return perturbed_action_trans, perturbed_action_quat_xyzw, pcd


# ----- for getting ground-truth translation action -----

def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy, device = pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx, device = pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6

    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt, device = hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


# ----- for getting discrete rot_grip_coll action in fill_replay() and get one-hot ground-truth action -----

def quaternion_to_discrete_euler(quaternion, resolution, gimble_fix=True):
    """
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    """
    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


# ----- for getting discretize translation, rotation, gripper open, and ignore collision actions in fill_replay() -----

def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],             
    rotation_resolution: int,           
    crop_augmentation: bool,            
):
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


# ----- for adding noise to image w.r.t. wpt_local -----

def add_uni_noi(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x


# used only in train.py
def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


# ========================= functions only used in eval mode ========================= 

def select_feat_from_hm_cache(
    pt_cam: torch.Tensor,
    hm: torch.Tensor,
    pt_cam_wei: torch.Tensor,
) -> torch.Tensor:
    """
    Cached version of select_feat_from_hm where we feed in directly the
    intermediate value of pt_cam and pt_cam_wei. Look into the original
    function to get the meaning of these values and return type. It could be
    used while inference if the location of the points remain the same.
    """

    nc, nw, h, w = hm.shape
    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, -1, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler("xyz", euluer, degrees=True).as_quat()


# ========================= functions only used in two-stage scheme =========================

# adjust the pc according to loc position and scale 
def trans_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        if len(pc.shape) == 2:
            pc = sca * (pc - loc)
        else:
            pc = sca * (pc - loc.unsqueeze(1))
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans