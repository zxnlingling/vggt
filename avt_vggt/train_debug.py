import os
import time
import yaml
import tqdm
import torch
import wandb
import argparse

from models import avt_vggt
import torch.distributed as dist
from vggt_agent import VGGTAgent
from collections import defaultdict
from contextlib import redirect_stdout
from configs import vggt_config as vggt_cfg_mod
from configs import vggt_exp_config as exp_cfg_mod
from utils.log_utils import get_logdir, print_loss_log
from utils.vggt_utils import DATA_FOLDER, get_model_para
from utils.mvt_utils import get_num_feat, PreprocessAgent2
from utils.env_utils import COLOSSEUM_TASKS, RLBENCH_TASKS
from waypoint_extraction.select_keyframe import get_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

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

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch


def dump_log(exp_cfg, vggt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/vggt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(vggt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


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


def train(agent, dataset, training_iterations, log_iter, rank=0, node_rank=0, epoch=0, ifwandb=False):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):
        # if rank == 0:
        #     print("Start timing for one iteration ...")
        # t_start_0 = time.time()

        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        
        update_args = {
            "step": iteration,
            "rank": rank,
            "epoch": epoch
        }
        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )

        # t_end_0 = time.time()
        # if rank == 0:
        #     print("Prepared replay samples and update args of this batch. Time Cost: {} minutes".format((t_end_0 - t_start_0) / 60.0))

        return_out = agent.update(**update_args)
        if (iteration + 1) % 100 == 0 and rank == 0:
            # t_start_log = time.time()

            loss_log = agent.loss_log
            total_loss_avg = sum(loss_log['total_loss'][-100:]) / len(loss_log['total_loss'][-100:])
            trans_loss_avg = sum(loss_log['trans_loss'][-100:]) / len(loss_log['trans_loss'][-100:])

            print(f"total loss: {total_loss_avg} | trans loss: {trans_loss_avg}")

            if ifwandb and node_rank == 0:
                wandb.log(data = {
                                    'total_loss': loss_log['total_loss'][iteration],
                                    'trans_loss': loss_log['trans_loss'][iteration],
                                    'rot_loss_x': loss_log['rot_loss_x'][iteration],
                                    'rot_loss_y': loss_log['rot_loss_y'][iteration],
                                    'rot_loss_z': loss_log['rot_loss_z'][iteration],
                                    'grip_loss': loss_log['grip_loss'][iteration],
                                    'collision_loss': loss_log['collision_loss'][iteration],
                                    'lr': loss_log['lr'][iteration],
                                    }, 
                            step = log_iter)
            
            # t_end_log = time.time()
            # print("Updated loss log for one iteration. Time Cost: {} minutes".format((t_end_log - t_start_log) / 60.0))
              
        log_iter += 1
        torch.cuda.empty_cache()
        
    if rank == 0:
        log = print_loss_log(agent)

    return log


def experiment(cmd_args, rank, node_rank, world_size):

    # torch.cuda.set_per_process_memory_fraction(0.5)  
    device = f"cuda:{rank % world_size}"

    ddp = world_size > 1

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))
    if ddp:
        print(f"Running DDP on rank {rank}.")
    old_exp_cfg_peract_lr = exp_cfg.vggt.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.vggt.lr *= world_size * exp_cfg.bs
    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    EPOCHS = exp_cfg.epochs
    NUM_TRAIN = exp_cfg.demo
    BATCH_SIZE_TRAIN = exp_cfg.bs
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * world_size))

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
    log_dir = get_logdir(cmd_args, exp_cfg)

    if rank == 0:
        print("Training on {} tasks: {}".format(len(tasks), tasks))
    
    vggt_cfg = vggt_cfg_mod.get_cfg_defaults()
    if cmd_args.vggt_cfg_path != "":
        vggt_cfg.merge_from_file(cmd_args.vggt_cfg_path)
    if cmd_args.vggt_cfg_opts != "":
        vggt_cfg.merge_from_list(cmd_args.vggt_cfg_opts.split(" "))
    vggt_cfg.feat_dim = get_num_feat(exp_cfg.rvt)
    vggt_cfg.freeze()
    # for maintaining backward compatibility
    assert vggt_cfg.num_rot == exp_cfg.rvt.num_rotation_classes, print(
        vggt_cfg.num_rot, exp_cfg.rvt.num_rotation_classes
    )

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks=tasks,
        BATCH_SIZE_TRAIN=BATCH_SIZE_TRAIN,
        BATCH_SIZE_TEST=None,
        TRAIN_REPLAY_STORAGE_DIR=TRAIN_REPLAY_STORAGE_DIR,             
        TEST_REPLAY_STORAGE_DIR=None,
        DATA_FOLDER=DATA_FOLDER,                            
        NUM_TRAIN=NUM_TRAIN,
        NUM_VAL=None,
        refresh_replay=cmd_args.refresh_replay,
        device=device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,      
    )
    train_dataset = get_dataset_func()
    t_end = time.time()

    if rank == 0:
        print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    vggt = avt_vggt.AVT_VGGT(
        renderer_device=device,
        rank=rank,
        **vggt_cfg,
    ).to(device)

    if rank == 0:
        get_model_para(vggt)
    if ddp:
        vggt = DDP(vggt, device_ids=[device], find_unused_parameters=True) # gradient_as_bucket_view=True

    agent = VGGTAgent(
        network=vggt,
        cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
        log_dir=f"{log_dir}/test_run/",
        **exp_cfg.rvt,
        **exp_cfg.vggt,
    )
    agent.build(training=True, device=device)

    start_epoch = 0
    end_epoch = EPOCHS

    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume

        if rank == 0:
            print(f"Recovering model and checkpoint from {exp_cfg.resume}")

        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1

    elif os.path.exists(f'{log_dir}/model_last.pth'):
        
        agent_path = f'{log_dir}/model_last.pth'
        if rank == 0:
            print(f"resume from checkpoint")
        
        epoch = load_agent(agent_path, agent, only_epoch=False)
        if rank == 0:
            print(f"Recovering model and checkpoint from {agent_path}, model epoch: {epoch}")
        start_epoch = epoch + 1
        
    dist.barrier()

    if exp_cfg.wandb and rank == 0 and node_rank == 0:
        mode = os.getenv("WANDB_MODE", "online")
        #key = os.getenv("WANDB_API_KEY")
        if mode != "offline":
            wandb.login(key="22442aa42bfa49b8803c69c971721ecb80912f16")
        
        wandb.init(
            project=exp_cfg.exp_id,
            name=exp_cfg.exp_name,
            config=exp_cfg,
            save_code=False
        )

    if rank == 0:
        # logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.vggt.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.vggt.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, vggt_cfg, cmd_args, log_dir)
        exp_cfg.vggt.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()

    if rank == 0:
        print("Start training ...", flush=True)

    i = start_epoch
    log_iter = 0
    while True:
        if i == end_epoch:
            break

        if rank == 0:
            print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")

        out = train(agent, train_dataset, TRAINING_ITERATIONS, log_iter, rank, node_rank, epoch=i, ifwandb=exp_cfg.wandb)

        # TODO: eval here

        if rank == 0 and node_rank == 0:
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
            # if os.path.exists(f'{log_dir}/model_{i-1}.pth'):
            #     os.remove(f'{log_dir}/model_{i-1}.pth')
        
        i += 1
        log_iter += TRAINING_ITERATIONS

    if rank == 0:
        print("[Finish]")



if __name__ == "__main__":

    # command: NCCL_ALGO=Ring torchrun --nproc_per_node="8" --nnodes="1" --master_port=30002 train_debug.py --exp_cfg_opts "tasks all_rlbench" 
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False) 
    parser.add_argument("--vggt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")
    parser.add_argument("--vggt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="debug_runs")

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    # Set the URL for communication
    dist_url = "env://" # default

    # Retrieve world_size and rank
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])

    # Initialize the process group
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
 
    try:
        # Pass the LOCAL_RANK to the experiment function
        local_rank = int(os.environ["LOCAL_RANK"])
        experiment(cmd_args, local_rank, rank, world_size)  
    except Exception as e:
        import traceback
        print(f"Process {rank} failed with error: {e}")
        traceback.print_exc()

    finally:
        os._exit(-1)