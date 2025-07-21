import math
import wandb
import numpy as np
from scipy.spatial.transform import Rotation

EPS = np.finfo(float).eps * 4.0

def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    E.g.:
    >>> q0 = random_quat()
    >>> q1 = random_quat()
    >>> q = quat_slerp(q0, q1, 0.0)
    >>> np.allclose(q, q0)
    True

    >>> q = quat_slerp(q0, q1, 1.0)
    >>> np.allclose(q, q1)
    True

    >>> q = quat_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2.0, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2.0, math.acos(-np.dot(q0, q1)) / angle)
    True

    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def linear_interpolation(p1, p2, t):
    """Compute the linear interpolation between two 3D points"""
    return p1 + t * (p2 - p1)


def point_line_distance(point, line_start, line_end):
    """Compute the shortest distance between a 3D point and a line segment defined by two 3D points"""
    line_vector = line_end - line_start
    point_vector = point - line_start
    # t represents the position of the orthogonal projection of the given point onto the infinite line defined by the segment
    t = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)
    t = np.clip(t, 0, 1)
    projection = linear_interpolation(line_start, line_end, t)
    return np.linalg.norm(point - projection)


def point_quat_distance(point, quat_start, quat_end, t, total):
    pred_point = quat_slerp(quat_start, quat_end, fraction=t / total)
    err_quat = (
        Rotation.from_quat(pred_point) * Rotation.from_quat(point).inv()
    ).magnitude()
    return err_quat


def geometric_waypoint_trajectory(actions, gt_states, waypoints, return_list=False):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints
    gt_pos = [p["robot0_eef_pos"] for p in gt_states]
    gt_quat = [p["robot0_eef_quat"] for p in gt_states]

    keypoints_pos = [actions[k][:3] for k in waypoints]
    keypoints_quat = [gt_quat[k] for k in waypoints]

    state_err = []

    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]
        start_keypoint_quat = keypoints_quat[i]
        end_keypoint_quat = keypoints_quat[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_pos[start_idx:end_idx]
        segment_points_quat = gt_quat[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            rot_err = point_quat_distance(
                segment_points_quat[i],
                start_keypoint_quat,
                end_keypoint_quat,
                i,
                len(segment_points_quat),
            )
            state_err.append(pos_err + rot_err)

    # print the average and max error for pos and rot
    # print(f"Average pos error: {np.mean(pos_err_list):.6f} \t Average rot error: {np.mean(rot_err_list):.6f}")
    # print(f"Max pos error: {np.max(pos_err_list):.6f} \t Max rot error: {np.max(rot_err_list):.6f}")

    if return_list:
        return total_traj_err(state_err), state_err
    return total_traj_err(state_err)


def pos_only_geometric_waypoint_trajectory(
    actions, gt_states, waypoints, return_list=False
):
    """Compute the geometric trajectory from the waypoints"""

    # prepend 0 to the waypoints for geometric computation
    if waypoints[0] != 0:
        waypoints = [0] + waypoints

    keypoints_pos = [actions[k] for k in waypoints]
    state_err = []
    n_segments = len(waypoints) - 1

    for i in range(n_segments):
        # Get the current keypoint and the next keypoint
        start_keypoint_pos = keypoints_pos[i]
        end_keypoint_pos = keypoints_pos[i + 1]

        # Select ground truth points within the current segment
        start_idx = waypoints[i]
        end_idx = waypoints[i + 1]
        segment_points_pos = gt_states[start_idx:end_idx]

        # Compute the shortest distances between ground truth points and the current segment
        for i in range(len(segment_points_pos)):
            pos_err = point_line_distance(
                segment_points_pos[i], start_keypoint_pos, end_keypoint_pos
            )
            state_err.append(pos_err)

    # print the average and max error
    print(
        f"Average pos error: {np.mean(state_err):.6f} \t Max pos error: {np.max(state_err):.6f}"
    )

    if return_list:
        return total_traj_err(state_err), state_err
    else:
        return total_traj_err(state_err)


def reconstruct_waypoint_trajectory(
    env,
    actions,
    gt_states,
    waypoints,
    initial_state=None,
    video_writer=None,
    verbose=True,
    remove_obj=False,
):
    """Reconstruct the trajectory from a set of waypoints"""
    curr_frame = 0
    pred_states_list = []
    traj_err_list = []
    frames = []
    total_DTW_distance = 0

    if len(gt_states) <= 1:
        return pred_states_list, traj_err_list, 0

    # reset the environment
    # reset_env(env=env, initial_state=initial_state, remove_obj=remove_obj)
    env._task.reset_to_demo(initial_state)
    prev_k = 0

    for k in waypoints:
        # skip to the next keypoint
        if verbose:
            print(f"Next keypoint: {k}")
        _, _, _, info = env.step(actions[k], skip=k - prev_k, record=True)
        prev_k = k

        # select a subset of states that are the closest to the ground truth
        DTW_distance, pred_states = dynamic_time_warping(
            gt_states[curr_frame + 1 : k + 1], info["intermediate_states"]
        )
        if verbose:
            print(
                f"selected subsequence: {pred_states} with DTW distance: {DTW_distance}"
            )
        total_DTW_distance += DTW_distance
        for i in range(len(pred_states)):
            # measure the error between the recorded and actual states
            curr_frame += 1
            pred_state_idx = pred_states[i]
            err_dict = compute_state_error(
                gt_states[curr_frame], info["intermediate_states"][pred_state_idx]
            )
            traj_err_list.append(total_state_err(err_dict))
            pred_states_list.append(
                info["intermediate_states"][pred_state_idx]["robot0_eef_pos"]
            )
            # save the frames (agentview)

    # wandb log the video if initialized
    if video_writer is not None and wandb.run is not None:
        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        wandb.log({"video": wandb.Video(video, fps=20, format="mp4")})

    # compute the total trajectory error
    traj_err = total_traj_err(traj_err_list)
    if verbose:
        print(
            f"Total trajectory error: {traj_err:.6f} \t Total DTW distance: {total_DTW_distance:.6f}"
        )

    return pred_states_list, traj_err_list, traj_err


def total_state_err(err_dict):
    return err_dict["err_pos"] + err_dict["err_quat"]


def total_traj_err(err_list):
    # return np.mean(err_list)
    return np.max(err_list)


def compute_state_error(gt_state, pred_state):
    """Compute the state error between the ground truth and predicted states."""
    err_pos = np.linalg.norm(gt_state["robot0_eef_pos"] - pred_state["robot0_eef_pos"])
    err_quat = (
        Rotation.from_quat(gt_state["robot0_eef_quat"])
        * Rotation.from_quat(pred_state["robot0_eef_quat"]).inv()
    ).magnitude()
    err_joint_pos = np.linalg.norm(
        gt_state["robot0_joint_pos"] - pred_state["robot0_joint_pos"]
    )
    state_err = dict(err_pos=err_pos, err_quat=err_quat, err_joint_pos=err_joint_pos)
    return state_err


def dynamic_time_warping(seq1, seq2, idx1=0, idx2=0, memo=None):
    if memo is None:
        memo = {}

    if idx1 == len(seq1):
        return 0, []

    if idx2 == len(seq2):
        return float("inf"), []

    if (idx1, idx2) in memo:
        return memo[(idx1, idx2)]

    distance_with_current = total_state_err(compute_state_error(seq1[idx1], seq2[idx2]))
    error_with_current, subseq_with_current = dynamic_time_warping(
        seq1, seq2, idx1 + 1, idx2 + 1, memo
    )
    error_with_current += distance_with_current

    error_without_current, subseq_without_current = dynamic_time_warping(
        seq1, seq2, idx1, idx2 + 1, memo
    )

    if error_with_current < error_without_current:
        memo[(idx1, idx2)] = error_with_current, [idx2] + subseq_with_current
    else:
        memo[(idx1, idx2)] = error_without_current, subseq_without_current

    return memo[(idx1, idx2)]