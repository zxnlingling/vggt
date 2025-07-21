# Code originally copied from https://github.com/lucys0/awe
# Waypoint-Based Imitation Learning for Robotic Manipulation (Conference on Robot Learning 2023)
""" Automatic waypoint selection """

import copy
import numpy as np
from typing import List
from .traj_reconstruction import pos_only_geometric_waypoint_trajectory, reconstruct_waypoint_trajectory, geometric_waypoint_trajectory


""" Iterative waypoint selection """
def greedy_waypoint_selection(
    actions=None,           # 动作序列，长度等于 demo 的帧数
    gt_states=None,
    err_threshold=None,
    pos_only=False
):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    if not pos_only:
        for i in range(len(actions) - 1):
            if actions[i][-1] != actions[i + 1][-1]:
                waypoints.append(i)
                waypoints.append(i + 1)
        waypoints.sort()

    # reconstruct the trajectory, and record the reconstruction error for each state
    for i in range(len(actions)):
        func = (
            pos_only_geometric_waypoint_trajectory
            if pos_only
            else geometric_waypoint_trajectory
        )
        total_traj_err, reconstruction_error = func(
            actions=actions,
            gt_states=gt_states,
            waypoints=waypoints,
            return_list=True,
        )

        # break if the reconstruction error is below the threshold
        if total_traj_err < err_threshold:
            break
        # add the frame of the highest reconstruction error as a waypoint, excluding frames that are already waypoints
        max_error_frame = np.argmax(reconstruction_error)
        while max_error_frame in waypoints:
            reconstruction_error[max_error_frame] = 0
            max_error_frame = np.argmax(reconstruction_error)
        waypoints.append(max_error_frame)
        waypoints.sort()

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


def _is_stopped(demo, i, stopped_buffer, delta):
    obs = demo[i]
    next_is_final = i == (len(demo) - 2)
    gripper_state_no_change = (                                         # no change in the continuous 4 frames (excluding the last frame)
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)      # joint velocities approach 0 with absolute tolerance delta
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_final) and gripper_state_no_change)
    return stopped


""" Heuristic waypoint selection """
def heuristic_waypoint_selection(actions=None, gt_states=None, err_threshold=0.1, stopped_buffer=4, demo=None):
    # make the last frame a waypoint
    waypoints = [len(actions) - 1]

    # make the frames of gripper open/close waypoints
    for i in range(len(actions) - 1):
        if actions[i][-1] != actions[i + 1][-1]:
            waypoints.append(i + 1)
    waypoints.sort()
    
    stopped_buffer = 0
    for i in range(len(gt_states)):
        # choose one from 4 continuouly static frames after moving
        stopped = _is_stopped(demo, i, stopped_buffer, delta=err_threshold) 
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        if i != 0 and i != 1 and stopped:
            waypoints.append(i)            
        
    waypoints = list(dict.fromkeys(waypoints))  
    waypoints.sort()
    if len(waypoints) > 1 and (waypoints[-1] - 1) == waypoints[-2]:
        waypoints.pop(-2)                                           # remove the second to last frame if included
    print("=======================================================================")
    print(f"Selected {len(waypoints)} waypoints: {waypoints}")
    return waypoints


""" Time-Linear Interpolation based on Heuristic waypoint selection """
def fixed_number_waypoint_selection(actions=None, gt_states=None, demo=None, method=None, num_keypoints=20):
    waypoints = []
    if method == 'heuristic':
        basic_waypoints = heuristic_waypoint_selection(actions=actions, gt_states=gt_states, demo=demo)
        waypoints = interpolated_keypoint(len(demo), basic_waypoints, num_keypoints)
        return waypoints

    elif method == 'random':
        waypoints = np.random.choice(
            range(len(demo)),
            size=num_keypoints,
            replace=False)
        waypoints.sort() # ascending order
        return waypoints

    elif method == 'fixed_interval':
        segment_length = len(demo) // num_keypoints
        for i in range(0, len(demo), segment_length): # choose 20 keypoints with fixed interval
            waypoints.append(i)
        return waypoints

    else:
        raise NotImplementedError


def interpolated_keypoint(total_steps: int, keypoint_step: List[int], num_keypoints: int) -> List[int]:
    if not keypoint_step:
        return []
    # 初始化结果列表，并确保包含keypoint_step中的点
    new_steps = list(set(keypoint_step.copy()))  # 去重初始关键点   # keypoint_step.copy()
    max_index = total_steps - 1  # 最大索引
    if max_index <= 0 or num_keypoints <= 1:
        return sorted(new_steps)
    # 每隔多少个点插入一个关键点
    sample_interval = max_index / (num_keypoints - 1)
    # 处理每个关键点区间
    for i in range(len(keypoint_step) - 1):
        start = keypoint_step[i]
        end = keypoint_step[i + 1]
        interval_length = end - start
        if interval_length <= 0:
            continue
        # 计算该区间需要插入的点数，并采用线性（均匀）插值
        num_insert = int(interval_length // sample_interval)      ##结果可能为负数，但如果处理的都是正数可以不计，考虑max(0, int(math.floor(interval_length / sample_interval)))
        if num_insert > 0:
            insert_steps = [
                start + int((end - start) * (j / (num_insert + 1)))
                for j in range(1, num_insert + 1)
            ]
            new_steps.extend(insert_steps)

    # 处理最后一个关键点到终点的区间
    last_start = keypoint_step[-1]
    last_end = max_index
    last_length = last_end - last_start

    if last_length > 0:
        num_insert = int(last_length // sample_interval)
        if num_insert > 0:
            insert_steps = [
                last_start + int((last_end - last_start) * (j / (num_insert + 1)))
                for j in range(1, num_insert + 1)
            ]
            new_steps.extend(insert_steps)

    # 去重、按照时间步长排序并限制数量
    new_steps = sorted(set(new_steps))
    while len(new_steps) > num_keypoints:
        new_steps.pop()  # 从末尾逐步移除多余点

    return new_steps


""" Backtrack waypoint selection """
def backtrack_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):
    # add heuristic waypoints
    num_frames = len(actions)

    # make the last frame a waypoint
    waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            waypoints.append(i)
    waypoints.sort()

    # backtracing to find the optimal waypoints
    start = 0
    while start < num_frames - 1:
        for end in range(num_frames - 1, 0, -1):
            rel_waypoints = [k - start for k in waypoints if k >= start and k < end] + [
                end - start
            ]
            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[start : end + 1],
                gt_states=gt_states[start + 1 : end + 2],
                waypoints=rel_waypoints,
                verbose=False,
                initial_state=initial_states[start],
                remove_obj=remove_obj,
            )
            if total_traj_err < err_threshold:
                waypoints.append(end)
                waypoints = list(set(waypoints))
                waypoints.sort()
                break
        start = end

    print("=======================================================================")
    print(
        f"Selected {len(waypoints)} waypoints: {waypoints} \t total trajectory error: {total_traj_err:.6f}"
    )
    return waypoints


""" DP waypoint selection """
# use geometric interpretation
def dp_waypoint_selection(
    actions=None,
    gt_states=None,
    err_threshold=None,
    pos_only=False,
):
    if actions is None:
        actions = copy.deepcopy(gt_states)
    elif gt_states is None:
        gt_states = copy.deepcopy(actions)
        
    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    # if not pos_only:
    for i in range(num_frames - 1):
        if actions[i][-1] != actions[i + 1][-1]:
            initial_waypoints.append(i)
            # initial_waypoints.append(i + 1)
    initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])
    func = (
        pos_only_geometric_waypoint_trajectory
        if pos_only
        else geometric_waypoint_trajectory
    )

    # Check if err_threshold is too small, then return all points as waypoints
    min_error = func(actions, gt_states, list(range(1, num_frames)))
    if err_threshold < min_error:
        print("Error threshold is too small, returning all points as waypoints.")
        return list(range(1, num_frames))

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            total_traj_err = func(
                actions=actions[k : i + 1],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
            )

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(
        f"Minimum number of waypoints: {len(waypoints)} \tTrajectory Error: {total_traj_err}"
    )
    print(f"waypoint positions: {waypoints}")

    return waypoints


def dp_reconstruct_waypoint_selection(
    env, actions, gt_states, err_threshold, initial_states, remove_obj
):

    num_frames = len(actions)

    # make the last frame a waypoint
    initial_waypoints = [num_frames - 1]

    # make the frames of gripper open/close waypoints
    for i in range(num_frames - 1):
        if actions[i, -1] != actions[i + 1, -1]:
            initial_waypoints.append(i)
    initial_waypoints.sort()

    # Memoization table to store the waypoint sets for subproblems
    memo = {}

    # Initialize the memoization table
    for i in range(num_frames):
        memo[i] = (0, [])

    memo[1] = (1, [1])

    # Populate the memoization table using an iterative bottom-up approach
    for i in range(1, num_frames):
        min_waypoints_required = float("inf")
        best_waypoints = []

        for k in range(1, i):
            # waypoints are relative to the subsequence
            waypoints = [j - k for j in initial_waypoints if j >= k and j < i] + [i - k]

            _, _, total_traj_err = reconstruct_waypoint_trajectory(
                env=env,
                actions=actions[k - 1 : i],
                gt_states=gt_states[k : i + 1],
                waypoints=waypoints,
                verbose=False,
                initial_state=initial_states[k - 1],
                remove_obj=remove_obj,
            )

            print(f"i: {i}, k: {k}, total_traj_err: {total_traj_err}")

            if total_traj_err < err_threshold:
                subproblem_waypoints_count, subproblem_waypoints = memo[k - 1]
                total_waypoints_count = 1 + subproblem_waypoints_count

                if total_waypoints_count < min_waypoints_required:
                    min_waypoints_required = total_waypoints_count
                    best_waypoints = subproblem_waypoints + [i]

                    print(
                        f"min_waypoints_required: {min_waypoints_required}, best_waypoints: {best_waypoints}"
                    )

        memo[i] = (min_waypoints_required, best_waypoints)

    min_waypoints_count, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    # remove duplicates
    waypoints = list(set(waypoints))
    waypoints.sort()
    print(f"Minimum number of waypoints: {len(waypoints)}")
    print(f"waypoint positions: {waypoints}")

    return waypoints