import os
import csv
import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libffi.so.7"

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from torch.nn.parallel import DistributedDataParallel as DDP
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.backend.exceptions import BoundaryError, WaypointError, TaskEnvironmentError, NoWaypointsError, DemoError, InvalidActionError

import vggt_agent
from models.avt_vggt import AVT_VGGT
from vggt_agent import VGGTAgent as Agent
from utils.vggt_utils import get_eval_parser
import configs.vggt_config as default_vggt_cfg
import configs.vggt_exp_config as default_exp_cfg
from utils.custom_colosseum_env import CustomColosseumEnv
from utils.env_utils import CustomMultiTaskRLBenchEnv2 as Env
from utils.env_utils import (SimpleAccumulator, create_obs_config, SCENE_BOUNDS, CAMERAS, IMAGE_SIZE, 
                              EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning, ReplayTransition,
                              RLBENCH_TASKS, COLOSSEUM_TASKS, VideoSummary, ActResult)


class RolloutGenerator(object):

    def __init__(self, env_device = 'cuda:0'):
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, rank, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False):            
        
        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()

        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

        for step in range(episode_length):
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            
            if not replay_ground_truth:
                act_result = agent.act(step, rank, prepped_data, deterministic=eval)
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])
 
            agent_obs_elems = {k: np.array(v) for k, v in act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}

            transition = env.step(act_result)

            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step, rank, prepped_data, deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition
            if transition.info.get("needs_reset", transition.terminal):
                return
         

class RolloutGenerator_colosseum(object):

    def __init__(self, env_device): # env_device = 'cuda:0'
        self._env_device = env_device

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, env: CustomColosseumEnv, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0, # i.e. current_variation_id
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False,
                  task_name: str = "",current_rollout: int = 0):    
        
        if eval:
            obs = env.eval_reset_to_demo(task_name, eval_demo_seed, current_rollout)
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()

        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

        for ep_step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            if not replay_ground_truth:
                act_result = agent.act(prepped_data, deterministic=eval)
            else:
                if ep_step >= len(actions):
                    return
                act_result = ActResult(actions[ep_step])

            agent_obs_elems = {k: np.array(v) for k, v in act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in act_result.replay_elements.items()}
            transition = env.step(act_result)

            obs_tp1 = dict(transition.observation)
            timeout = False
            if ep_step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(prepped_data, deterministic=eval)

                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in act_result.observation_elements.items()}

                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or ep_step == episode_length - 1:
                env._action_mode.arm_action_mode.record_end(env._task._scene, steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
            

def load_agent_only_model(agent_path, agent=None, only_epoch=False):

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in vggt1. "
                    "Be cautious if you are using a two stage network."
                )
                model.vggt1.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)

    return epoch
            

def load_agent(
    model_path=None,
    exp_cfg_path=None,
    vggt_cfg_path=None,
    eval_log_dir="",
    device=0,
):

    assert model_path is not None

    # load exp_cfg
    model_folder = os.path.join(os.path.dirname(model_path))

    exp_cfg = default_exp_cfg.get_cfg_defaults()
    if exp_cfg_path != None:
        exp_cfg.merge_from_file(exp_cfg_path)
    else:
        exp_cfg.merge_from_file(os.path.join(model_folder, "exp_cfg.yaml"))
    exp_cfg.freeze()

    # create agent
    vggt_cfg = default_vggt_cfg.get_cfg_defaults()
    if vggt_cfg_path != None:
        vggt_cfg.merge_from_file(vggt_cfg_path)
    else:
        vggt_cfg.merge_from_file(os.path.join(model_folder, "vggt_cfg.yaml"))
    vggt_cfg.freeze()

    EPOCHS = exp_cfg.epochs
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * 8))

    vggt = AVT_VGGT(
        renderer_device=f"cuda:{device}",
        rank=device,
        **vggt_cfg,
    )

    agent = vggt_agent.VGGTAgent(
        network=vggt.to(f"cuda:{device}"),
        cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{eval_log_dir}/eval_run",
        **exp_cfg.vggt,
        **exp_cfg.rvt,
    )
    agent.build(training=False, device=device)
    load_agent_only_model(model_path, agent)
    agent.eval()

    print("Agent Information")
    print(agent)
    return agent


@torch.no_grad()
def eval(
    rank, 
    agent,
    tasks,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    replay_ground_truth=False,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    verbose=True,
    save_video=False,
):
    agent.eval()
    if isinstance(agent, vggt_agent.VGGTAgent):
        agent.load_clip()

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = create_obs_config(CAMERAS, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all_rlbench":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)
    elif tasks[0] == "all_colosseum":
        tasks = COLOSSEUM_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    eval_env = Env(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=1 if save_video else -1,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    if logging:
        assert log_dir is not None

        # create metric saving writer
        csv_file = "eval_results.csv"
        if not os.path.exists(os.path.join(log_dir, csv_file)):
            with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_writer.writeheader()

    # evaluate agent
    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                rank=rank, 
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=replay_ground_truth,
            )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except (RuntimeError, IndexError, BoundaryError, WaypointError, NoWaypointsError, DemoError, InvalidActionError, TaskEnvironmentError) as e:
                print(f"Evaluating {tasks[task_id]} | Episode {ep} | Error: " + str(e))
                eval_env.shutdown()
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]
        if logging:
            # writer csv first
            with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
                fieldnames = ["task", "success rate", "length", "total_transitions"]
                csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
                csv_results = {"task": task_name}
                for s in summaries:
                    if s.name == "eval_envs/return":
                        csv_results["success rate"] = s.value
                    elif s.name == "eval_envs/length":
                        csv_results["length"] = s.value
                    elif s.name == "eval_envs/total_transitions":
                        csv_results["total_transitions"] = s.value
                    if "eval" in s.name:
                        s.name = "%s/%s" % (s.name, task_name)
                csv_writer.writerow(csv_results)
        else:
            for s in summaries:
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)

        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        scores.append(task_score)

        if save_video:
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
            os.makedirs(record_folder, exist_ok=True)
            video_success_cnt = 0
            video_fail_cnt = 0
            video_cnt = 0
            target_width, target_height = 1280, 720

            for summary in summaries:
                if isinstance(summary, VideoSummary):
                    video = deepcopy(summary.value)
                    video = np.transpose(video, (0, 2, 3, 1))
                    video = video[:, :, :, ::-1]
                    if task_rewards[video_cnt] > 99:
                        video_path = os.path.join(
                            record_folder,
                            f"{task_name}_success_{video_success_cnt}.mp4",
                        )
                        video_success_cnt += 1
                    else:
                        video_path = os.path.join(
                            record_folder, f"{task_name}_fail_{video_fail_cnt}.mp4"
                        )
                        video_fail_cnt += 1

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
                    writer = cv2.VideoWriter(
                        video_path, 
                        fourcc,
                        record_fps,
                        (target_width, target_height)
                    )

                    if not writer.isOpened():
                        print(f"Video writer not supported.")

                    for idx in range(len(video)):
                        try:
                            frame = cv2.resize(video[idx], (target_width, target_height), 
                                                interpolation=cv2.INTER_CUBIC)
                            writer.write(frame)
                        except Exception as e:
                                print(f"Video frame processing failed: {str(e)}")
                                continue
                    
                    writer.release()
                    video_cnt += 1
                    

    eval_env.shutdown()

    if logging:
        with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
            fieldnames = ["task", "success rate", "length", "total_transitions"]
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_results = {"task": "average"}
            csv_results["success rate"] = sum(scores) / len(scores)
            csv_writer.writerow(csv_results)
        csv_fp.close()

    # set agent to back train mode
    agent.train()

    # unloading clip to save memory
    if isinstance(agent, vggt_agent.VGGTAgent):
        agent.unload_clip()
        agent._network.free_mem()

    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    model_paths = []
    assert args.model_name is not None
    model_paths.append(os.path.join(args.model_folder, args.model_name))

    for model_path in model_paths:
        tasks_to_eval = deepcopy(args.tasks)

        model_idx = get_model_index(model_path)
        if model_idx is None:
            model_idx = 0

        agent = load_agent(
            model_path=model_path,
            exp_cfg_path=args.exp_cfg_path,
            vggt_cfg_path=args.vggt_cfg_path,
            eval_log_dir=args.eval_log_dir,
            device=args.device,
        )

        agent_eval_log_dir = os.path.join(
            args.eval_log_dir, os.path.basename(model_path).split(".")[0]
        )
        os.makedirs(agent_eval_log_dir, exist_ok=True)

        scores = eval(
            rank=args.device, 
            agent=agent,
            tasks=tasks_to_eval,
            eval_datafolder=args.eval_datafolder,
            start_episode=args.start_episode,
            eval_episodes=args.eval_episodes,
            episode_length=args.episode_length,
            replay_ground_truth=args.ground_truth,
            device=args.device,
            headless=args.headless,
            logging=True,
            log_dir=agent_eval_log_dir,
            verbose=True,
            save_video=args.save_video,
        )
        print(f"model {model_path}, scores {scores}")
        task_scores = {}
        for i in range(len(tasks_to_eval)):
            task_scores[tasks_to_eval[i]] = scores[i]

        print("save ", task_scores)


if __name__ == "__main__":

    parser = get_eval_parser()
    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = "none"

    args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    os.makedirs(args.eval_log_dir, exist_ok=True)
    # save the arguments for future reference
    with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
        yaml.dump(args.__dict__, fp)

    _eval(args)