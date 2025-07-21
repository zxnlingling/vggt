import os
import pickle
import numpy as np
from PIL import Image
from clip import tokenize
from functools import reduce
from multiprocessing import Lock
from typing import Any, List, Type
from pyrep.const import RenderMode
from abc import ABC, abstractmethod
from rlbench.backend.task import Task
from pyrep.objects import VisionSensor, Dummy
from rlbench.backend.observation import Observation
from rlbench.backend.utils import image_to_float_array
from rlbench.action_modes.action_mode import ActionMode
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from rlbench import ObservationConfig, Environment, CameraConfig
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, Scene


# ========================= constant ========================= 

IMAGE_SIZE = 128
IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
CAMERA_FRONT = 'front'
CAMERA_WRIST = 'wrist'
DEPTH_SCALE = 2**24 - 1
IMAGE_FORMAT  = '%d.png'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
EPISODE_FOLDER = "episode%d"
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"  # the pkl file that contains language goals for each demonstration
CAMERAS = ["front", "left_shoulder", "right_shoulder", "wrist"]
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6,]    
REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces', 'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces', 'task_low_dim_state', 'misc']
# tasks
RLBENCH_TASKS = [
    "put_item_in_drawer",           
    "reach_and_drag",               
    "turn_tap",                     
    "slide_block_to_color_target",  
    "open_drawer",                  
    "put_groceries_in_cupboard",    
    "place_shape_in_shape_sorter",  
    "put_money_in_safe",            
    "push_buttons",                 
    "close_jar",                    
    "stack_blocks",                 
    "place_cups",                   
    "place_wine_at_rack_location",  
    "light_bulb_in",                
    "sweep_to_dustpan_of_size",     
    "insert_onto_square_peg",       
    "meat_off_grill",               
    "stack_cups",                   
]
COLOSSEUM_TASKS = [
    "basketball_in_hoop",
    "close_laptop_lid",
    "get_ice_from_fridge",
    "insert_onto_square_peg",
    "move_hanger",
    "place_wine_at_rack_location",
    "reach_and_drag",
    "setup_chess",
    "stack_cups",
    "turn_oven_on",
    "close_box",
    "empty_dishwasher",
    "hockey",
    "meat_on_grill",
    "open_drawer",
    "put_money_in_safe",
    "scoop_with_spatula",
    "slide_block_to_target",
    "straighten_rope",
    "wipe_desk"
]


# ========================= functions only used in fill_replay() ========================= 

# get demo 
def get_stored_demo(data_path, index):
    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)
  
    # low dim pickle file
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        obs = pickle.load(f)

    # variation number
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)

    num_steps = len(obs)
    for i in range(num_steps):
        # obs[i].ignore_collisions = np.array(1.)
        obs[i].front_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), 
                                                            IMAGE_FORMAT % i)))
        obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), 
                                                                    IMAGE_FORMAT % i))) 
        obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), 
                                                                     IMAGE_FORMAT % i)))
        obs[i].wrist_rgb = np.array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), 
                                                            IMAGE_FORMAT % i)))

        obs[i].front_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), 
                                                                          IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[i].front_depth = near + obs[i].front_depth * (far - near)

        obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_LS, 
                                                                                                           IMAGE_DEPTH), 
                                                                                                           IMAGE_FORMAT % i)), 
                                                                                                           DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
        obs[i].left_shoulder_depth = near + obs[i].left_shoulder_depth * (far - near)

        obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), 
                                                                                   IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
        obs[i].right_shoulder_depth = near + obs[i].right_shoulder_depth * (far - near)

        obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), 
                                                                          IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth, 
                                                                                    obs[i].misc['front_camera_extrinsics'],
                                                                                    obs[i].misc['front_camera_intrinsics'])
        obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].left_shoulder_depth, 
                                                                                            obs[i].misc['left_shoulder_camera_extrinsics'],
                                                                                            obs[i].misc['left_shoulder_camera_intrinsics'])
        obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].right_shoulder_depth, 
                                                                                             obs[i].misc['right_shoulder_camera_extrinsics'],
                                                                                             obs[i].misc['right_shoulder_camera_intrinsics'])
        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth, 
                                                                                           obs[i].misc['wrist_camera_extrinsics'],
                                                                                           obs[i].misc['wrist_camera_intrinsics'])
    return obs


# create obs_dict 
def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False,
                episode_length: int = 10):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([
        obs.gripper_open,
        *obs.gripper_joint_positions,
        *grip_pose,
        *joint_pos])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}
    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)
    
    for camera_name in cameras:
        obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]
    
    # add timestep to low_dim_state
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose
    return obs_dict


# ========================= arm_action_mode in eval() ========================= 

class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)


# ========================= Summary classes ========================= 

class Summary(object):
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

class HistogramSummary(Summary):
    pass

class ScalarSummary(Summary):
    pass

class ImageSummary(Summary):
    pass


# ========================= tools for SimpleAccumulator ========================= 

class ReplayTransition(object):
    def __init__(self, observation: dict, action: np.ndarray,
                 reward: float, terminal: bool, timeout: bool,
                 final_observation: dict = None,
                 summaries: List[Summary] = None,
                 info: dict = None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.timeout = timeout
        # final only populated on last timestep
        self.final_observation = final_observation
        self.summaries = summaries or []
        self.info = info
          

class StatAccumulator(object):

    def step(self, transition: ReplayTransition, eval: bool):
        pass

    def pop(self) -> List[Summary]:
        pass

    def peak(self) -> List[Summary]:
        pass

    def reset(self) -> None:
        pass


class Metric(object):

    def __init__(self):
        self._previous = []
        self._current = 0

    def update(self, value):
        self._current += value

    def next(self):
        self._previous.append(self._current)
        self._current = 0

    def reset(self):
        self._previous.clear()

    def min(self):
        return np.min(self._previous)

    def max(self):
        return np.max(self._previous)

    def mean(self):
        return np.mean(self._previous)

    def median(self):
        return np.median(self._previous)

    def std(self):
        return np.std(self._previous)

    def __len__(self):
        return len(self._previous)

    def __getitem__(self, i):
        return self._previous[i]
    

class _SimpleAccumulator(StatAccumulator):

    def __init__(self, prefix, eval_video_fps: int = 30,
                 mean_only: bool = True):
        self._prefix = prefix
        self._eval_video_fps = eval_video_fps
        self._mean_only = mean_only
        self._lock = Lock()
        self._episode_returns = Metric()
        self._episode_lengths = Metric()
        self._summaries = []
        self._transitions = 0

    def _reset_data(self):
        with self._lock:
            self._episode_returns.reset()
            self._episode_lengths.reset()
            self._summaries.clear()

    def step(self, transition: ReplayTransition, eval: bool):
        with self._lock:
            self._transitions += 1
            self._episode_returns.update(transition.reward)
            self._episode_lengths.update(1)
            if transition.terminal:
                self._episode_returns.next()
                self._episode_lengths.next()
            self._summaries.extend(list(transition.summaries))

    def _get(self) -> List[Summary]:
        sums = []

        if self._mean_only:
            stat_keys = ["mean"]
        else:
            stat_keys = ["min", "max", "mean", "median", "std"]
        names = ["return", "length"]
        metrics = [self._episode_returns, self._episode_lengths]
        for name, metric in zip(names, metrics):
            for stat_key in stat_keys:
                if self._mean_only:
                    assert stat_key == "mean"
                    sum_name = '%s/%s' % (self._prefix, name)
                else:
                    sum_name = '%s/%s/%s' % (self._prefix, name, stat_key)
                sums.append(
                    ScalarSummary(sum_name, getattr(metric, stat_key)()))
        sums.append(ScalarSummary(
            '%s/total_transitions' % self._prefix, self._transitions))
        sums.extend(self._summaries)
        return sums

    def pop(self) -> List[Summary]:
        data = []
        if len(self._episode_returns) > 0:
            data = self._get()
            self._reset_data()
        return data

    def peak(self) -> List[Summary]:
        return self._get()
    
    def reset(self):
        self._transitions = 0
        self._reset_data()


class SimpleAccumulator(StatAccumulator):

    def __init__(self, eval_video_fps: int = 30, mean_only: bool = True):
        self._train_acc = _SimpleAccumulator(
            'train_envs', eval_video_fps, mean_only=mean_only)
        self._eval_acc = _SimpleAccumulator(
            'eval_envs', eval_video_fps, mean_only=mean_only)

    def step(self, transition: ReplayTransition, eval: bool):
        if eval:
            self._eval_acc.step(transition, eval)
        else:
            self._train_acc.step(transition, eval)

    def pop(self) -> List[Summary]:
        return self._train_acc.pop() + self._eval_acc.pop()

    def peak(self) -> List[Summary]:
        return self._train_acc.peak() + self._eval_acc.peak()
    
    def reset(self) -> None:
        self._train_acc.reset()
        self._eval_acc.reset()


# ========================= tools for RLBench env ========================= 

# create observation_config for RLBench env
def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def change_case(str):
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, str).lower()


ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']


def _extract_obs(obs: Observation, channels_last: bool, observation_config, grip_pose, joint_pos):
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = np.array([ # different from yarr.envs to adapt to 18-D proprio input
        obs.gripper_open,
        *obs.gripper_joint_positions,
        *grip_pose,
        *joint_pos])
    # Remove all of the individual state elements
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS}
    if not channels_last:
        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # Add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, 'left_shoulder'),
        (observation_config.right_shoulder_camera, 'right_shoulder'),
        (observation_config.front_camera, 'front'),
        (observation_config.wrist_camera, 'wrist'),
        (observation_config.overhead_camera, 'overhead')]:
        if config.point_cloud:
            obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
            obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    return obs_dict


class ObservationElement(object):

    def __init__(self, name: str, shape: tuple, type: Type[np.dtype]):
        self.name = name
        self.shape = shape
        self.type = type


def _get_cam_observation_elements(camera: CameraConfig, prefix: str, channels_last):
    elements = []
    img_s = list(camera.image_size)
    shape = img_s + [3] if channels_last else [3] + img_s
    if camera.rgb:
        elements.append(
            ObservationElement('%s_rgb' % prefix, shape, np.uint8))
    if camera.point_cloud:
        elements.append(
            ObservationElement('%s_point_cloud' % prefix, shape, np.float32))
        elements.append(
            ObservationElement('%s_camera_extrinsics' % prefix, (4, 4),
                               np.float32))
        elements.append(
            ObservationElement('%s_camera_intrinsics' % prefix, (3, 3),
                               np.float32))
    if camera.depth:
        shape = img_s + [1] if channels_last else [1] + img_s
        elements.append(
            ObservationElement('%s_depth' % prefix, shape, np.float32))
    if camera.mask:
        raise NotImplementedError()

    return elements


def _observation_elements(observation_config, channels_last) -> List[ObservationElement]:
    elements = []
    robot_state_len = 0
    if observation_config.joint_velocities:
        robot_state_len += 7
    if observation_config.joint_positions:
        robot_state_len += 7
    if observation_config.joint_forces:
        robot_state_len += 7
    if observation_config.gripper_open:
        robot_state_len += 1
    if observation_config.gripper_pose:
        robot_state_len += 7
    if observation_config.gripper_joint_positions:
        robot_state_len += 2
    if observation_config.gripper_touch_forces:
        robot_state_len += 2
    if observation_config.task_low_dim_state:
        raise NotImplementedError()
    if robot_state_len > 0:
        elements.append(ObservationElement(
            'low_dim_state', (robot_state_len,), np.float32))
    elements.extend(_get_cam_observation_elements(
        observation_config.left_shoulder_camera, 'left_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.right_shoulder_camera, 'right_shoulder', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.front_camera, 'front', channels_last))
    elements.extend(_get_cam_observation_elements(
        observation_config.wrist_camera, 'wrist', channels_last))
    return elements


class Transition(object):

    def __init__(self, observation: dict, reward: float, terminal: bool,
                 info: dict = None, summaries: List[Summary] = None):
        self.observation = observation
        self.reward = reward
        self.terminal = terminal
        self.info = info or {}
        self.summaries = summaries or []


class TextSummary(Summary):
    pass


class VideoSummary(Summary):
    def __init__(self, name: str, value: Any, fps: int = 30):
        super(VideoSummary, self).__init__(name, value)
        self.fps = fps


class ActResult(object):

    def __init__(self, action: Any,
                 observation_elements: dict = None,
                 replay_elements: dict = None,
                 info: dict = None):
        self.action = action
        self.observation_elements = observation_elements or {}
        self.replay_elements = replay_elements or {}
        self.info = info or {}


class Env(ABC):

    def __init__(self):
        self._active_task_id = 0
        self._eval_env = False

    @property
    def eval(self):
        return self._eval_env

    @eval.setter
    def eval(self, eval):
        self._eval_env = eval

    @property
    def active_task_id(self) -> int:
        return self._active_task_id

    @abstractmethod
    def launch(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        pass

    @property
    @abstractmethod
    def observation_elements(self) -> List[ObservationElement]:
        pass

    @property
    @abstractmethod
    def action_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        pass


class MultiTaskEnv(Env):

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        pass


class MultiTaskRLBenchEnv(MultiTaskEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 channels_last=False,
                 headless=True,
                 swap_task_every: int = 1,
                 include_lang_goal_in_obs=False):
        super(MultiTaskRLBenchEnv, self).__init__()
        self._task_classes = task_classes
        self._observation_config = observation_config
        self._channels_last = channels_last
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._rlbench_env = Environment(
            action_mode=action_mode, obs_config=observation_config,
            dataset_root=dataset_root, headless=headless)
        self._task = None
        self._task_name = ''
        self._lang_goal = 'unknown goal'
        self._swap_task_every = swap_task_every
        self._rlbench_env
        self._episodes_this_task = 0
        self._active_task_id = -1

        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}

    def _set_new_task(self, shuffle=False):
        if shuffle:
            self._active_task_id = np.random.randint(0, len(self._task_classes))
        else:
            self._active_task_id = (self._active_task_id + 1) % len(self._task_classes)
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = self._task_classes[self._active_task_id]
        self._task = self._rlbench_env.get_task(task)

        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def extract_obs(self, obs: Observation, grip_pose, joint_pos):
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config, grip_pose, joint_pos)
        if self._include_lang_goal_in_obs:
            extracted_obs['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def launch(self):
        self._rlbench_env.launch()
        self._set_new_task()

    def shutdown(self):
        self._rlbench_env.shutdown()

    def reset(self) -> dict:
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        descriptions, obs = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant
        extracted_obs = self.extract_obs(obs)

        return extracted_obs

    def step(self, action: np.ndarray) -> Transition:
        obs, reward, terminal = self._task.step(action)
        obs = self.extract_obs(obs)
        return Transition(obs, reward, terminal)

    @property
    def observation_elements(self) -> List[ObservationElement]:
        return _observation_elements(self._observation_config, self._channels_last)

    @property
    def action_shape(self):
        return (self._rlbench_env.action_size, )

    @property
    def env(self) -> Environment:
        return self._rlbench_env

    @property
    def num_tasks(self) -> int:
        return len(self._task_classes)
    

class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 swap_task_every: int = 1,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes, observation_config, action_mode, dataset_root,
            channels_last, headless=headless, swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  
                # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):

        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs, grip_pose, joint_pos)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}',
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # super(CustomMultiTaskRLBenchEnv, self).reset()

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
    

class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict