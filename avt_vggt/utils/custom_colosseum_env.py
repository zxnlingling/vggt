import os
import json
import numpy as np
from clip import tokenize
from pyrep.const import RenderMode
from rlbench.backend.task import Task
from omegaconf import DictConfig, OmegaConf
from typing import Type, List, Optional, Any
from pyrep.objects import VisionSensor, Dummy
from colosseum.rlbench.utils import name_to_class
from rlbench.backend.observation import Observation
from rlbench.action_modes.action_mode import ActionMode
from rlbench.observation_config import ObservationConfig
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError, TaskEnvironmentError
from colosseum.rlbench.extensions.environment import EnvironmentExt, DEFAULT_PATH_TTMS
from colosseum.tools.dataset_generator import should_collect_task, get_spreadsheet_config
from colosseum import ASSETS_CONFIGS_FOLDER, ASSETS_JSON_FOLDER, TASKS_PY_FOLDER, TASKS_TTM_FOLDER
from .env_utils import CAMERAS, REMOVE_KEYS, ActResult, VideoSummary, TextSummary, Transition, change_case


def get_colosseum_cfg(task, variation_id):
    base_cfg_path = os.path.join(ASSETS_CONFIGS_FOLDER, f"{task}.yaml")
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, 'r') as f:
            base_cfg = OmegaConf.load(f)
    collection_cfg_path: str = (
        os.path.join(ASSETS_JSON_FOLDER, base_cfg.env.task_name) + ".json"
    )
    collection_cfg: Optional[Any] = None
    with open(collection_cfg_path, "r") as fh:
        collection_cfg = json.load(fh)
    if collection_cfg is None:
        return 1
    if "strategy" not in collection_cfg:
        return 1
    idx_to_collect = (                                      # Check if the user wants to collect all variations (-1) or only one
        base_cfg.data.idx_to_collect
        if "idx_to_collect" in base_cfg.data
        else -1
    )
    if should_collect_task(collection_cfg, variation_id, idx_to_collect):
        config = get_spreadsheet_config(base_cfg, collection_cfg, variation_id)
        return config


class CustomColosseumEnv(EnvironmentExt):
    def __init__(
        self,
        task_classes: List[Type[Task]],
        obs_config: ObservationConfig,
        action_mode: ActionMode,
        episode_length: int,
        dataset_root: str = "",
        headless: bool = False,
        swap_task_every: int = 1,
        time_in_state: bool = False,
        include_lang_goal_in_obs: bool = False,
        record_every_n: int = 20,
        path_task_ttms: str = DEFAULT_PATH_TTMS,
        reward_scale=100.0,
        env_config: DictConfig = DictConfig({}),
        ):
        super().__init__(action_mode)
        self._task = None
        self._task_classes = task_classes
        self._obs_config = obs_config
        self._action_mode = action_mode
        self._episode_length = episode_length
        self._dataset_root = dataset_root
        self._headless = headless
        self._swap_task_every = swap_task_every
        self._time_in_state = time_in_state
        self._include_lang_goal_in_obs = include_lang_goal_in_obs
        self._record_every_n = record_every_n
        self._path_task_ttms = path_task_ttms  
        self._env_config = env_config  
        self._record_cam = None
        self._lang_goal = 'unknown goal'
        self._i = 0
        self._recorded_images = []
        self._active_task_id = -1
        self._task_name_to_idx = {change_case(tc.__name__):i for i, tc in enumerate(self._task_classes)}
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None
        self._episode_index = 0
        self._reward_scale = reward_scale

    @property
    def eval(self):
        return self._eval_env

    @eval.setter
    def eval(self, eval):
        self._eval_env = eval

    def shutdown(self):
        self._rlbench_env.shutdown()

    @property
    def active_task_id(self) -> int:
        return self._active_task_id

    def set_task(self, task_name: str):
        self._active_task_id = self._task_name_to_idx[task_name]
        task = name_to_class(task_name, TASKS_PY_FOLDER)
        self._task = self._rlbench_env.get_task(task)
        descriptions, _ = self._task.reset()
        self._lang_goal = descriptions[0] # first description variant

    def set_variation_colosseum(self, v: int) -> None:
        task_name = self._task.get_name()
        valid_variation_ids = set()
        for _, dirs, _ in os.walk(self._dataset_root):
            for dir_name in dirs:
                if dir_name.startswith(task_name):
                    if '_' not in dir_name:
                        continue
                    *_, dir_id = dir_name.split('_')
                    try:
                        valid_variation_ids.add(int(dir_id))
                    except ValueError:
                        continue
            if v not in valid_variation_ids:
                raise TaskEnvironmentError(
                    f'Variation ID {v} not found. Valid IDs are: {sorted(valid_variation_ids)}')
        self._task._variation_number = v

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def launch(self, task, env_cfg):
        # np.random.seed(None)
        self._rlbench_env = EnvironmentExt(
            action_mode = self._action_mode,
            obs_config = self._obs_config,
            headless = self._headless,
            path_task_ttms = TASKS_TTM_FOLDER, # dataset_root=self._dataset_root,
            env_config = env_cfg,
        )

        self._rlbench_env.launch()
        self.set_task(task)

        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([1280, 720])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)


    def extract_obs(self, obs: Observation, cameras, t=None, prev_action=None, channels_last: bool = False):
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
        # NOTE: The key 'ignore_collisions' does not exist in test dataset of COLOSSEUM.
        obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32) 
        for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
            obs_dict[k] = v.astype(np.float32)
        
        for camera_name in cameras:
            obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
            obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

        if self._include_lang_goal_in_obs:
            obs_dict['lang_goal_tokens'] = tokenize([self._lang_goal])[0].numpy()

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)
        obs.gripper_matrix = grip_mat
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        return obs_dict

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
            obs = self.extract_obs(obs, CAMERAS)
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

    def eval_reset_to_demo(self, task_name, variation_id, current_rollout):
        self._i = 0
        demo = self.get_demos(amount=1, image_paths=True, variation_number=-1, task_name=task_name, 
                              random_selection=False, from_episode_number=current_rollout)[0]   

        if "place_wine_at_rack_location" not in task_name:
            self.set_variation_colosseum(variation_id)
        obs_tmp = demo[0]

        desc, obs = self._task.reset_to_demo(demo)
        self._lang_goal = desc[0] 

        # Now check that the state has been properly restored
        np.testing.assert_allclose(
            obs.joint_positions, obs_tmp.joint_positions, atol=1e-1)
        np.testing.assert_allclose(
            obs.gripper_open, obs_tmp.gripper_open, atol=1e-1)

        self._previous_obs_dict = self.extract_obs(obs, CAMERAS)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
    
    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))