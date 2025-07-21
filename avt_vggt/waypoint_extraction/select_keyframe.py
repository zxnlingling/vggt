import os
import re
import clip
import torch
import shutil
import pickle
import logging
import numpy as np
from typing import List

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))  

from utils.dataset_utils import ObservationElement, ReplayElement, ReplayBuffer, UniformReplayBuffer, PyTorchReplayBuffer
from utils.mvt_utils import _get_action, _clip_encode_text, LOW_DIM_SIZE, VOXEL_SIZES, DEMO_AUGMENTATION_EVERY_N, ROTATION_RESOLUTION
from .extract_waypoints import greedy_waypoint_selection, heuristic_waypoint_selection, dp_waypoint_selection, fixed_number_waypoint_selection
from utils.env_utils import COLOSSEUM_TASKS, IMAGE_SIZE, SCENE_BOUNDS, VARIATION_DESCRIPTIONS_PKL, EPISODE_FOLDER, CAMERAS, get_stored_demo, extract_obs


ERROR_THRESHOLD = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]  


def create_replay(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = 3 + 1
    gripper_pose_size = 7
    ignore_collisions_size = 1
    max_token_seq_len = 77
    lang_feat_dim = 1024
    lang_emb_dim = 512

    # low_dim_state
    observation_elements = []
    observation_elements.append(
        ObservationElement("low_dim_state", (LOW_DIM_SIZE,), np.float32)
    )

    # rgb, depth, point cloud, intrinsics, extrinsics
    for cname in cameras:
        observation_elements.append(
            ObservationElement(
                "%s_rgb" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_depth" % cname,
                (
                    1,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_point_cloud" % cname,
                (
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                np.float32,
            )
        )  # see pyrep/objects/vision_sensor.py on how pointclouds are extracted from depth frames
        observation_elements.append(
            ObservationElement(
                "%s_camera_extrinsics" % cname,
                (
                    4,
                    4,
                ),
                np.float32,
            )
        )
        observation_elements.append(
            ObservationElement(
                "%s_camera_intrinsics" % cname,
                (
                    3,
                    3,
                ),
                np.float32,
            )
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings
    observation_elements.extend(
        [
            ReplayElement("trans_action_indicies", (trans_indicies_size,), np.int32),
            ReplayElement(
                "rot_grip_action_indicies", (rot_and_grip_indicies_size,), np.int32
            ),
            ReplayElement("ignore_collisions", (ignore_collisions_size,), np.int32),
            ReplayElement("gripper_pose", (gripper_pose_size,), np.float32),
            ReplayElement(
                "lang_goal_embs",
                (
                    max_token_seq_len,
                    lang_emb_dim,
                ),  # extracted from CLIP's language encoder
                np.float32,
            ),
            ReplayElement(
                "lang_goal", (1,), object
            ),  # language goal string for debugging and visualization
        ]
    )

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
        ReplayElement("total_keypoints", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )

    return replay_buffer


def fill_replay(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
    keyframe_method=None
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)

    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        print("Filling replay ...:", task)
        for d_idx in range(start_idx, start_idx + num_demos): # from 0 to 99
            print("Filling demo %d" % d_idx)
            demo = get_stored_demo(data_path=data_path, index=d_idx)
            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, episode_folder % d_idx, variation_desriptions_pkl
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)

            # extract keyframes
            actions = []
            gt_states = []
            positions = []
            for k in range(len(demo)):            
                obs_tp1 = demo[k]  # 获取第k步的观测
                grip = float(obs_tp1.gripper_open)
                action = np.concatenate([obs_tp1.gripper_pose, np.array([grip])])
                actions.append(action)
                eef_pos = obs_tp1.gripper_pose[:3]      # Tensor shape=(3,)
                eef_quat = obs_tp1.gripper_pose[3:]     # Tensor shape=(4,)
                joint_pos = obs_tp1.joint_positions     # Tensor shape=(7,)
                vel_ang = obs_tp1.joint_velocities[3:5] # Tensor shape=(3,), End-effector angular velocity
                vel_lin = obs_tp1.joint_velocities[:3]  # Tensor shape=(3,), End-effector cartesian velocity
                positions.append(eef_pos)
                gt_states.append(dict(robot0_eef_pos=eef_pos,
                                    robot0_eef_quat=eef_quat,
                                    robot0_joint_pos=joint_pos,
                                    robot0_vel_ang=vel_ang,
                                    robot0_vel_lin=vel_lin,))

            task_index = COLOSSEUM_TASKS.index(task)

            if keyframe_method == "heuristic":          # basic
                episode_keypoints = heuristic_waypoint_selection(actions=actions, gt_states=gt_states, demo=demo)
            elif keyframe_method == "greedy_pose":      # only position error 
                episode_keypoints = greedy_waypoint_selection(actions=positions, gt_states=positions, 
                                                              err_threshold=ERROR_THRESHOLD[task_index], pos_only=True)
            elif keyframe_method == "greedy_geometric": # position + quat + gripper
                episode_keypoints = greedy_waypoint_selection(actions=actions, gt_states=gt_states, 
                                                              err_threshold=ERROR_THRESHOLD[task_index], pos_only=False)
            elif keyframe_method == "dp_pose":          # global minimum number of keyframes that match position error
                episode_keypoints = dp_waypoint_selection(actions=actions, gt_states=actions, # actions=positions, gt_states=positions,
                                                          err_threshold=ERROR_THRESHOLD[task_index], pos_only=True)
            elif keyframe_method == "dp_geometric":     # global minimum number of keyframes that match geometric error
                episode_keypoints = dp_waypoint_selection(actions=actions, gt_states=gt_states, 
                                                          err_threshold=ERROR_THRESHOLD[task_index], pos_only=False)
            elif keyframe_method == "fixed_number":
                episode_keypoints = fixed_number_waypoint_selection(actions=actions, gt_states=gt_states, demo=demo, method='heuristic', num_keypoints=10)

            else:
                env = create_env_from_demo(env_meta=demo, task_name=f"{task}_0") 
                env.shutdown()

            next_keypoint_idx = 0
            for i in range(len(demo) - 1):
                if not demo_augmentation and i > 0:
                    break
                if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
                    continue

                obs = demo[i]
                desc = descs[0]
                # if our starting point is past one of the keypoints, then remove it
                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break

                # add individual data points to a replay
                prev_action = None
                for k in range(next_keypoint_idx, len(episode_keypoints)):
                    keypoint = episode_keypoints[k]
                    obs_tp1 = demo[keypoint] # Observation             
                    obs_tm1 = demo[max(0, keypoint - 1)]    # 前一帧 
                    # action是gripper_pose+gripper_open（float格式
                    (trans_indicies, rot_grip_indicies, _, action, _,) = _get_action(obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, 
                                                                                     rotation_resolution, crop_augmentation,)
                    terminal = k == len(episode_keypoints) - 1    # if is the last keypoint, terminal
                    reward = float(terminal) * 1.0 if terminal else 0
                    obs_dict = extract_obs(obs, CAMERAS, t=k - next_keypoint_idx, prev_action=prev_action, episode_length=25,)
                    tokens = clip.tokenize([desc]).numpy()
                    token_tensor = torch.from_numpy(tokens, device = device)
                    with torch.no_grad():
                        lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
                    obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()
                    prev_action = np.copy(action)
                    if k == 0:
                        keypoint_frame = -1
                    else:
                        keypoint_frame = episode_keypoints[k - 1]
                    others = {
                        "demo": True,
                        "keypoint_idx": k,
                        "episode_idx": d_idx,
                        "keypoint_frame": keypoint_frame,
                        "next_keypoint_frame": keypoint,
                        "sample_frame": i,
                        "total_keypoints": len(episode_keypoints)
                        }
                    final_obs = {
                        "trans_action_indicies": trans_indicies,
                        "rot_grip_action_indicies": rot_grip_indicies,  # rot + grip: 3+1
                        "gripper_pose": obs_tp1.gripper_pose,           # 3+4
                        "lang_goal": np.array([desc], dtype=object),
                        }
                    others.update(final_obs)
                    others.update(obs_dict)
                    timeout = False
                    replay.add(task, task_replay_storage_folder, action, reward, terminal, timeout, **others)
                    obs = obs_tp1

                obs_dict_tp1 = extract_obs(                             # final step
                    obs_tp1,
                    CAMERAS,
                    t=k + 1 - next_keypoint_idx,
                    prev_action=prev_action,
                    episode_length=25,
                )
                obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

                obs_dict_tp1.pop("wrist_world_to_cam", None)
                obs_dict_tp1.update(final_obs)
                replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ] : replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train=True,
    sample_distribution_mode="transition_uniform",
):

    train_replay_buffer = create_replay(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    for task in tasks:  
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        # EPISODES_FOLDER_TRAIN = f"replay_train_originInterpolate/test/{task}/variation0/episodes"
        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"

        if refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")

        fill_replay(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
            keyframe_method = "dp_pose"
        )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()
    return train_dataset


def create_env_from_demo(
    env_meta,
    task_name: str
):
    """
    Create environment.

    """
    from colosseum import TASKS_TTM_FOLDER
    from rlbench.backend import task as rlbench_task
    from colosseum.rlbench.utils import ObservationConfigExt
    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from ..custom_colosseum_env import CustomColosseumEnv, get_colosseum_cfg
    from ..utils.env_utils import EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning
    EVALFOLDER="/fs-computility/efm/shared/datasets/Official_Manipulation_Data/sim/colosseum/replay_train_originInterpolate/test/"
    match = re.fullmatch(r"(.+?)_(\d+)$", task_name)         
    if match:
        current_task = match.group(1)
    config = get_colosseum_cfg(current_task, 0)
    data_cfg, env_cfg = config.data, config.env
    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]
    task_classes = []
    for task in COLOSSEUM_TASKS:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))
    env = CustomColosseumEnv(
        task_classes = task_classes,
        obs_config = ObservationConfigExt(data_cfg),
        action_mode = MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        headless = True,
        path_task_ttms = TASKS_TTM_FOLDER,
        dataset_root = EVALFOLDER,
        episode_length = 25,
        swap_task_every = 100,
        include_lang_goal_in_obs = True,
        time_in_state = True,
        record_every_n = -1,
    )
    env.eval = True
    env.launch(current_task, env_cfg)
    desc, _ = env._task.reset_to_demo(env_meta)
    env._lang_goal = desc[0]
    print("Created environment with name {}".format(task_name))
    return env