from yacs.config import CfgNode as CN

_C = CN() 

_C.bs = 16                   
_C.demo = 100               
_C.tasks = "all_colosseum"  
_C.wandb = False            
_C.epochs = 90                      # TODO: tune this
_C.resume = ""              
_C.exp_id = "vggtact"       
_C.exp_name = 'debug'       
_C.train_iter = 16 * 10000  
_C.num_workers = 5                              # number of dataloader workers, >= 0 
_C.sample_distribution_mode = 'task_uniform'    # 'transition_uniform' or 'task_uniform'

# RVT-2
_C.rvt = CN()
_C.rvt.num_rotation_classes = 72
_C.rvt.transform_augmentation = True
_C.rvt.transform_augmentation_rpy = [0.0, 0.0, 45.0]
_C.rvt.transform_augmentation_xyz = [0.125, 0.125, 0.125] 

# VGGT
_C.vggt = CN()
_C.vggt.lr = 1.25e-5 # 2.5e-5
_C.vggt.img_aug = 0.0               # TODO: whether to use vggt img aug or not
_C.vggt.stage_two = True
_C.vggt.use_input_pc = True         # TODO: test whether to use original point cloud or generate with vggt
_C.vggt.warmup_steps = 2000
_C.vggt.optimizer_type = "lamb"     # "adamw" 
_C.vggt.image_resolution = 224
_C.vggt.lambda_weight_l2 = 1e-4 

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()



# _C.peract.add_rgc_loss = True
# _C.peract.same_trans_aug_per_seq = False
# _C.rvt.place_with_mean = False