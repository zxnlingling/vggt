from yacs.config import CfgNode as CN

_C = CN()

_C.depth = 8            # depth of RVT-2 attention layers 
_C.num_rot = 72 
_C.attn_dim = 512   
_C.feat_dim = (72 * 3) + 2 + 2
_C.lang_dim = 512   
_C.lang_len = 77    
_C.img_aug_2 = 0.05     # add noise to image before stage 1
_C.activation = "lrelu" 
_C.attn_heads = 8   
_C.im_channels = 64 
_C.proprio_dim = 18 
_C.wpt_img_aug = 0.0    
_C.attn_dropout = 0.1   
_C.img_feat_dim = 3     
_C.attn_dim_head = 64   
_C.lora_finetune = False 
_C.st_wpt_loc_aug = 0.05# add noise to wpt_local before stage 2
_C.weight_tie_layers = False        
_C.st_wpt_loc_inp_no_noise = True   

# VGGT
_C.lora_r = 16 
_C.st_sca = 4           # used in trans_pc
_C.add_corr = True  
_C.img_size = 128   
_C.add_depth = True     # used in renderer
_C.norm_corr = True     # used in renderer
_C.stage_two = True 
_C.use_renderer = False # whether to use renderer in stage 1
_C.add_pixel_loc = True # used in renderer
_C.img_patch_size = 8   # TODO: 14
_C.flash_attention = True  
_C.render_view_num = 3
_C.use_ray_renderer = True 
_C.image_resolution = 224

_C.vggt_config = "configs/vggt_train.yaml"
_C.vggt_ckpt = '/fs-computility/efm/lvqi/projects/colosseum/SAM2Act/sam2Act_COLOSSEUM/third_libraries/vggt/model.pt'

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()







# _C.decoder_dropout = 0.0
# _C.final_dim = 64
# _C.pe_fix = True
# _C.inp_pre_pro = False
# _C.inp_pre_con = False
# _C.ifsep = False