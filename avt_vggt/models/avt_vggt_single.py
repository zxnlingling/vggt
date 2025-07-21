import time
import torch
import logging
import os
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.mvt_utils import add_uni_noi, select_feat_from_hm
from utils.vggt_utils import interpolate_features, get_max_3d_frm_hm_cube, DPTHead_Custom
from utils.network_utils import (Conv2DBlock, DenseBlock, cache_fn, PreNorm, Attention, LayerNorm2d, 
                                 FeedForward, Fusion_up, ConvexUpSample, FixedPositionalEncoding, act_layer)

class AVT_VGGT_Single(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        proprio_dim,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        img_patch_size,
        add_corr,
        norm_corr,
        add_pixel_loc,
        add_depth,
        render_view_num,
        use_renderer,
        use_ray_renderer,
        wpt_img_aug,
        flash_attention,
        num_rot,
        st_sca,
        st_wpt_loc_aug,
        st_wpt_loc_inp_no_noise,
        vggt_config,
        vggt_ckpt,
        lora_r,
        lora_finetune,
        rank,
        vggt,
        image_resolution,
        vggt_down_ratio=[1, 2, 4, 8],
        vggt_feat_dim=128,
        renderer_device="cuda:0",
        renderer=None,
        no_feat=False,
    ):
        """Arbitrary-View Transfomer based on Visual Geometry Grounded Transformer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param im_channels: intermediate channel size
        :param img_patch_size: intial patch size
        :param norm_corr: wether or not to normalize the correspondece values.
            this matters when pc is outide -1, 1 like for the two stage mvt
        :param render_view_num: 4 means front, left, right, top. 
                                3 means front, right, top. (the same as point renderer)
                                2 means front, top. 
                                1 means front.
        :param vggt: 
        :param use_ray_renderer: whether to use the ray renderer or not
        :param wpt_img_aug: how much noise is added to the wpt_img while
            training, expressed as a percentage of the image size
        :default to use learned convex upsampling
        :param flash_attention: whether to use flash attention or not
        :version of the rotation prediction network: xyz prediction dependent on one another
        :param num_rot: number of discrete rotations per axis

        """

        super().__init__()
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.proprio_dim = proprio_dim
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.attn_dropout = attn_dropout
        self.add_corr = add_corr
        self.norm_corr = norm_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.wpt_img_aug = wpt_img_aug
        self.use_renderer = use_renderer
        self.use_ray_renderer = use_ray_renderer
        self.num_rot = num_rot
        self.no_feat = no_feat
        self.render_view_num = render_view_num
        self.vggt_down_ratio = vggt_down_ratio
        self.vggt_feat_dim = vggt_feat_dim
        self.rank = rank
        self.vggt = vggt
        self.image_resolution = image_resolution
        # for name, param in self.vggt.named_parameters():
        #     if param.requires_grad:
        #         print(name)  # for debug: print LoRA params only
        self.feat_img_dim = 48
        self.vggt_img_dim = 48
        self.curr_obs_idx = 0
        self.act_horizon = 1
        self.renderer = renderer

        if no_feat: # the first stage of two
            self.num_img = 4 # len(CAMERAS) of RLBench
        else:
            self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  

        # 64 img features + 64 proprio features
        self.input_dim_before_seq = self.im_channels * 2

        # learnable positional encoding
        lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        num_pe_token = spatial_size**2 * self.num_img

        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder (identity)
        self.input_preprocess = lambda x: x
        inp_pre_out_dim = inp_img_feat_dim

        # proprio preprocessing encoder
        self.proprio_preprocess = DenseBlock(
            self.proprio_dim,
            32, # rvt-2: self.im_channels
            norm="group",
            activation=activation,
        )
        
        # Extract features from tokens (based on DPT architecture)
        # feature_maps has shape (B, V, C, H//down_ratio, W//down_ratio) 
        self.feature_extractor_net = DPTHead_Custom(
            dim_in=2048,
            patch_size=self.vggt.aggregator.patch_size,
            features=self.vggt_feat_dim,
            feature_only=True,  # Only output features, no activation
            down_ratio=self.vggt_down_ratio[0],  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,           
            self.feat_img_dim,          
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # self.depth_patchify = Conv2DBlock(
        #     1,           
        #     self.feat_img_dim // 2,          
        #     kernel_sizes=self.vggt.aggregator.patch_size,
        #     strides=self.vggt.aggregator.patch_size,
        #     norm="group",
        #     activation=activation,
        #     padding=0,
        # )

        # lang preprocess
        self.lang_preprocess = DenseBlock(
            lang_emb_dim,
            self.im_channels * 2,
            norm="group",
            activation=activation,
        )
        self.refine_conv = torch.nn.Conv2d(self.vggt_feat_dim, self.vggt_feat_dim, kernel_size=3, stride=1, padding=1)
        if self.vggt_img_dim != 0:            
            self.fusion = Fusion_up(
                in_channels=128, 
                out_channels=self.vggt_img_dim, 
                kernel_sizes=1,
                strides=1,
                norm=None,
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )
        
        # self.input_dim_before_seq = 128

        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
                use_fast=flash_attention,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        # multi-resolution upsampling
        self.up0 = torch.nn.Sequential(  
            # 16x16 → 16x16    
            LayerNorm2d(self.input_dim_before_seq),
            LayerNorm2d(self.vggt_feat_dim),                        
            act_layer(activation),
            # 16x16 → 32x32    
            ConvexUpSample(in_dim=self.vggt_feat_dim, out_dim=self.vggt_feat_dim, up_ratio=2),
            LayerNorm2d(self.vggt_feat_dim),                        
            act_layer(activation),
            # 32x32 → 64x64
            ConvexUpSample(in_dim=self.vggt_feat_dim, out_dim=self.vggt_feat_dim, up_ratio=2),
            LayerNorm2d(self.vggt_feat_dim),
            act_layer(activation),
            # 64x64 → 128x128
            ConvexUpSample(in_dim=self.vggt_feat_dim, out_dim=self.vggt_feat_dim, up_ratio=2), 
            LayerNorm2d(self.vggt_feat_dim),
            act_layer(activation),
            # 128x128 → 128x128
            torch.nn.Conv2d(in_channels=self.vggt_feat_dim, out_channels=1, kernel_size=1)
        )
        # up0 used in RVT-2:
        # self.up0 = ConvexUpSample(
        #     in_dim=self.input_dim_before_seq,
        #     out_dim=1,
        #     up_ratio=self.img_patch_size,
        #     )

        if not self.no_feat:
            feat_fc_dim = 0
            feat_fc_dim += self.input_dim_before_seq
            feat_fc_dim += self.input_dim_before_seq

            def get_feat_fc(
                _feat_in_size,
                _feat_out_size,
                _feat_fc_dim=feat_fc_dim,   
            ):
                """
                _feat_in_size: input feature size
                _feat_out_size: output feature size
                _feat_fc_dim: hidden feature size
                """
                layers = [
                    nn.Linear(_feat_in_size, _feat_fc_dim),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim, _feat_fc_dim // 2),
                    nn.ReLU(),
                    nn.Linear(_feat_fc_dim // 2, _feat_out_size),
                ]
                feat_fc = nn.Sequential(*layers)
                return feat_fc

            feat_out_size = feat_dim

            # rot_ver = 1
            assert self.num_rot * 3 <= feat_out_size
            feat_out_size_ex_rot = feat_out_size - (self.num_rot * 3)
            if feat_out_size_ex_rot > 0:
                self.feat_fc_ex_rot = get_feat_fc(
                    self.num_img * feat_fc_dim, feat_out_size_ex_rot
                )

            self.feat_fc_init_bn = nn.BatchNorm1d(self.num_img * feat_fc_dim)
            self.feat_fc_pe = FixedPositionalEncoding(
                self.num_img * feat_fc_dim, feat_scale_factor=1
            )
            self.feat_fc_x = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
            self.feat_fc_y = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)
            self.feat_fc_z = get_feat_fc(self.num_img * feat_fc_dim, self.num_rot)

        global select_feat_from_hm
    
    
    # def vggt_image_encoder_forward(self, net, imgs, rank): 
    #     aggregated_tokens_list, ps_idx = net.aggregator(imgs, return_attn=False)
    #     feature_maps, feature_map_16, feature_map_32, feature_map_64, feature_map_128 = \
    #         self.feature_extractor_net(aggregated_tokens_list, imgs, ps_idx)
    #     if self.no_feat:
    #         self.match_rgb_feats = feature_maps
    #     del aggregated_tokens_list
    #     return feature_maps, feature_map_16, feature_map_32, feature_map_64, feature_map_128#, depth, depth_conf
    def vggt_image_encoder_forward(self, net, imgs, rank):
        # --- 优化1: 输入预处理 ---
        if imgs.dtype != torch.float16:  # 强制半精度
            imgs = imgs.half()
        
        # --- 优化2: 内存高效注意力 ---
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            aggregated_tokens_list, ps_idx = net.aggregator(
                imgs, 
                return_attn=False,
                use_memory_efficient=True  # 假设模型支持
            )
        
        # --- 优化3: 预分配特征内存 ---
        B, V = imgs.shape[:2]
        feature_shapes = {
            '128': (B, V, 128, 224, 224),
            '16': (B, V, 128, 16, 16),
            '32': (B, V, 128, 32, 32),
            '64': (B, V, 128, 64, 64)
        }
        features = {k: torch.empty(*v, dtype=torch.half, device=imgs.device) 
                for k, v in feature_shapes.items()}
        
        # --- 优化4: 梯度检查点 ---
        def feature_extraction_wrapper(*args):
            return self.feature_extractor_net(*args)
        
        feature_maps = torch.utils.checkpoint.checkpoint(
            feature_extraction_wrapper,
            aggregated_tokens_list,
            imgs,
            ps_idx,
            use_reentrant=False,
            preserve_rng_state=True
        )
        
        # --- 优化5: 显存及时释放 ---
        del aggregated_tokens_list
        torch.cuda.empty_cache()  # 强制清理
        
        # --- 优化6: 结果重组 ---
        with torch.inference_mode():  # 禁用自动求导
            feature_maps = feature_maps.to(dtype=torch.half)
            return (
                feature_maps,
                features['16'],
                features['32'],
                features['64'],
                features['128']
            )
    
    def get_matching_feature(self, match_input_dict):
        rgbs = match_input_dict['rgbs']
        kp_1 = match_input_dict['kp_1']
        kp_2 = match_input_dict['kp_2']
        kp_3 = match_input_dict['kp_3']
        kp_4 = match_input_dict['kp_4']
        normalize = match_input_dict['normalize']
        
        resize_factor = torch.tensor([518 / rgbs.shape[-1], 518 / rgbs.shape[-2]], device=rgbs.device)
        
        result = self.match_rgb_feats

        features = result.permute(0, 1, 3, 4, 2)        # (B, V, H, W, 128)
        features = features.reshape(-1, *features.shape[2:])  # (B*V, H, W, 128)
        features = features.permute(0, 3, 1, 2)         # (B*V, 128, H, W)

        features = self.refine_conv(features)  

        feature_1 = features[0::4]  # 0,4,8,...
        feature_2 = features[1::4]  # 1,5,9,...
        feature_3 = features[2::4]  # 2,6,10,...
        feature_4 = features[3::4]  # 3,7,11,...
            
        def process_view_feature(feature, kp, img_size=518, patch_size=self.img_patch_size, 
                                 stride=self.img_patch_size, normalize=normalize):
            batch_size = len(kp)
            max_kp = max(k.shape[0] for k in kp)
            kp_padded = torch.zeros(batch_size, max_kp, 2, device=feature.device)
            kp_mask = torch.zeros(batch_size, max_kp, dtype=torch.bool, device=feature.device)
            
            for i, k in enumerate(kp):
                kp_padded[i, :k.shape[0]] = k * resize_factor
                kp_mask[i, :k.shape[0]] = True
            
            interpolated = interpolate_features(
                feature, kp_padded, h=img_size, w=img_size, 
                patch_size=patch_size, stride=stride, 
                normalize=False
            ).permute(0, 2, 1) # (B, max_N, C)
            interpolated = interpolated * kp_mask.unsqueeze(-1)
            
            if normalize:
                valid_interpolated = interpolated[kp_mask]
                if valid_interpolated.numel() > 0:
                    interpolated[kp_mask] = F.normalize(valid_interpolated, p=2, dim=-1)
            return interpolated

        feature_1 = process_view_feature(feature_1, kp_1)
        feature_2 = process_view_feature(feature_2, kp_2)
        feature_3 = process_view_feature(feature_3, kp_3)
        feature_4 = process_view_feature(feature_4, kp_4)
        return feature_1, feature_2, feature_3, feature_4


    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        wpt_local=None,
        rot_x_y=None,
        rank=0,
        intrinsics=None,
        extrinsics=None,
        iteration=0,
        epoch=0,
        match_input_dict=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, 9 or 10, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param wpt_local: tensor of shape (bs, 3)
        :param rot_x_y: (bs, 2)
        :param intrinsics: list of num_img tensors, each (bs, 3, 3)
        :param extrinsics: list of num_img tensors, each (bs, 4, 4)
        :param rev_trans: set of functions, transforming wpt_local to global

        """

        # t_start_vggt1 = time.time()
    
        bs, num_img, img_feat_dim, h, w = img.shape         
        assert num_img == self.num_img
        assert h == w == self.img_size

        img_raw = img.clone()

        img = img.view(bs * num_img, img_feat_dim, h, w)    
        d0 = self.input_preprocess(img)                     

        # use VGGT to encode rgb_img
        rgb_img = img_raw[:, :, 3:6, :, :]

        # t_end_vggt1 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Process before VGGT finished. Time Cost: {} minutes".format((t_end_vggt1 - t_start_vggt1) / 60.0))
        
        # resize for VGGT
        if rgb_img.shape[-1] != self.image_resolution:

            original_shape = rgb_img.shape
            rgb_img = rgb_img.view(-1, *rgb_img.shape[2:])  # Reshape to (B*V, 3, H, W)
            resized_img = F.interpolate(rgb_img, size=(self.image_resolution, self.image_resolution), 
                                        mode='bicubic', align_corners=False)
            resized_img = torch.clamp(resized_img, 0.0, 1.0)
            resized_img = resized_img.view(*original_shape[:2], *resized_img.shape[1:])
            rgb_img = resized_img               # (b, render_view_num, 3, 224, 224)  
        # t_end_vggt2 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Resize for VGGT finished. Time Cost: {} minutes".format((t_end_vggt2 - t_end_vggt1) / 60.0))

        with torch.cuda.amp.autocast(enabled=True):
        
            self.vggt_rgb_feats_all = [None] * 5
            # for i in range(len(self.vggt_down_ratio)):
            # t_start_vggt3 = time.time()
            # if rank == 0:
            #     print("[DEBUG] Start timing for VGGT image encoding ... ")
            # vggt_rgb_feats, vggt_feats_16, vggt_feats_32, vggt_feats_64, vggt_feats_128, depth, depth_conf = \
            vggt_rgb_feats, vggt_feats_16, vggt_feats_32, vggt_feats_64, vggt_feats_128 = \
                self.vggt_image_encoder_forward(self.vggt, rgb_img, rank)    
            height, width = vggt_rgb_feats.shape[-2:]  
            self.vggt_rgb_feats_all[0] = vggt_feats_16
            self.vggt_rgb_feats_all[1] = vggt_feats_32
            self.vggt_rgb_feats_all[2] = vggt_feats_64
            self.vggt_rgb_feats_all[3] = vggt_feats_128
            self.vggt_rgb_feats_all[-1] = vggt_rgb_feats.view(bs*num_img, self.vggt_feat_dim, height, width)
            # t_end_vggt3 = time.time()
            # if rank == 0:
            #     print(f"[DEBUG] VGGT encoding finished. Image Size = {height}.", "Time Cost: {} minutes".format((t_end_vggt3 - t_start_vggt3) / 60.0))
            # t_start_vggt4 = time.time()
            vggt_out = self.vggt_rgb_feats_all[-1]  

        # c 128 -> vggt_img_dim
        rgb_img = self.fusion(vggt_out)
        num_pat_img = h // self.img_patch_size
        rgb_img = (rgb_img.view(bs, num_img, self.vggt_img_dim, num_pat_img, num_pat_img).transpose(1, 2).clone()) 
        # t_end_vggt4 = time.time()
        # if rank == 0:
        #     print("[DEBUG] VGGT rgb features obtained. Time Cost: {} minutes".format((t_end_vggt4 - t_start_vggt4) / 60.0))

        feat_img = img_raw.view(bs * num_img, img_feat_dim, h, w)
        feat_img = self.patchify(feat_img)                  # Conv2DBlock   (b*v, feat_img_dim/2, 16, 16)
        # # temp: for visualize purpose
        # import matplotlib.pyplot as plt
        # from utils.vggt_utils import save_rgb_images, visualize_depth, save_depth_images
        # if iteration == 0:
        #     if self.no_feat:
        #         save_rgb_images(img_raw[:, :, 3:6, :, :], f"debug_runs/images/temp_stage2/{self.no_feat}", prefix="original", 
        #                         views=["front", "left_shoulder", "right_shoulder", "wrist"])
        #     else:
        #         save_rgb_images(img_raw[:, :, 3:6, :, :], f"debug_runs/images/temp_stage2/{self.no_feat}", prefix="original", 
        #                         views=["top", "front", "right"])
        #     pc_temp = img_raw[:, :, :3, :, :].cpu().numpy()
        #     save_dir_2d = f"debug_runs/images/temp_stage2/{self.no_feat}/point_clouds"
        #     os.makedirs(save_dir_2d, exist_ok=True)
        #     for i in range(min(5, pc_temp.shape[0])):  # 保存前5个样本前4个视角
        #         for view_idx in range(min(4, pc_temp.shape[1])):  
        #             pc_restore = pc_temp[i, view_idx].transpose(1, 2, 0)
        #             fig, ax = plt.subplots()
        #             ax.imshow((pc_restore * 0.5 + 0.5).clip(0, 1))  # 假设归一化到 [-1, 1]
        #             ax.axis('off')  
        #             save_path = os.path.join(save_dir_2d, f"sample{i}_view{view_idx}.png")
        #             plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        #             plt.close()

        # if iteration % 800 == 0:
        #     save_depth_dir = f"debug_runs/images/temp_stage2/{self.no_feat}/depth"
        #     save_depth_images(img_raw[:, :, 6, :, :], save_depth_dir, epoch, iteration, prefix="GT", 
        #                       views=["front", "left", "right", "wrist"])
        #     visualize_depth(resized_img, depth, depth_conf, epoch, iteration, save_dir=save_depth_dir)
            
        feat_img = (
            feat_img.view(
                bs,
                num_img,
                self.feat_img_dim,
                num_pat_img,
                num_pat_img,
            ).transpose(1, 2).clone())          
        _, _, _d, _h, _w = feat_img.shape
        # t_end_vggt5 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Original rgb features obtained. Time Cost: {} minutes".format((t_end_vggt5 - t_end_vggt4) / 60.0))

        p = self.proprio_preprocess(proprio)  
        p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)    # (b, 32, 4, 16, 16)
        # t_end_vggt6 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Proprio features obtained. Time Cost: {} minutes".format((t_end_vggt6 - t_end_vggt5) / 60.0))
        #     print("[DEBUG] Start timing for RVT-2 multi-view transformer ... ")
        ins = torch.cat([rgb_img, feat_img, p], dim=1)                              # (b, 48+48+32 = 128, 4, 16, 16)
        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")      
        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")    
        
        # add 3*16*16 learable pos encoding (dim=128) only to image tokens
        ins += self.pos_encoding                        

        # append language features as sequence
        num_lang_tok = 0
        l = self.lang_preprocess(
            lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
        )
        l = l.view(bs, self.lang_max_seq_len, -1)   
        num_lang_tok = l.shape[1]
        ins = torch.cat((l, ins), dim=1)                    # (b, 77 + 3*16*16 = 845, 128)

        # t_start_vggt7 = time.time()
        
        x = self.fc_bef_attn(ins)                           # DenseBlock    (b, 77 + 3*16*16 = 845, attn_dim = 512)

        lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]               

        # within image self attention
        imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)    
        for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
            imgx = self_attn(imgx) + imgx
            imgx = self_ff(imgx) + imgx
        imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)       # (b*v, 16*16, 512) -> (b, v*16*16, 512)
        x = torch.cat((lx, imgx), dim=1)                

        # cross attention
        for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:     
            x = self_attn(x) + x
            x = self_ff(x) + x

        # throwing away the language embeddings
        x = x[:, num_lang_tok:]                                             
        x = self.fc_aft_attn(x)                                             # (b, v*16*16, 512) -> (b, v*16*16, 128)          
        # t_end_vggt7 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Attention of the original RVT-2 Time Cost: {} minutes".format((t_end_vggt7 - t_start_vggt7) / 60.0))       

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  
        x = rearrange(x, "b ... d -> b d ...")                                  

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]               # for global max of x
        _feat = rearrange(_feat, 'b c n -> b (c n)')                             
        feat.append(repeat(_feat, f'b d -> b {1} d'))                       # len = 1, shape = (b, 1, 128*4])

        x = (                                                               # (b*v, 128, 16, 16)
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        # t_start_vggt8 = time.time()
        # trick to stablize mixed-percision training
        with torch.cuda.amp.autocast(enabled=False):
            ln0, ln1, act1, conv2, ln2, act2, conv3, ln3, act3, conv4, ln4, act4, conv5 = self.up0   

            # feat_s0 = self.vggt_rgb_feats_all[4].reshape(bs * num_img, 128, 224, 224).float()    # [32, 128, 224, 224]
            feat_s1 = self.vggt_rgb_feats_all[0].reshape(bs * num_img, 128, num_pat_img, num_pat_img).float() 
            feat_s2 = self.vggt_rgb_feats_all[1].reshape(bs * num_img, 128, num_pat_img*2, num_pat_img*2).float()     
            feat_s3 = self.vggt_rgb_feats_all[2].reshape(bs * num_img, 128, num_pat_img*4, num_pat_img*4).float()     
            feat_s4 = self.vggt_rgb_feats_all[3].reshape(bs * num_img, 128, num_pat_img*8, num_pat_img*8).float() 

            x = x.float()

            upscaled_embedding = act1(ln0(x) + ln1(feat_s1))                    # (b*v, 128, 16, 16)
            upscaled_embedding = act2(ln2(conv2(upscaled_embedding) + feat_s2)) # (b*v, 128, 32, 32)
            upscaled_embedding = act3(ln3(conv3(upscaled_embedding) + feat_s3)) # (b*v, 128, 64, 64)
            upscaled_embedding = act4(ln4(conv4(upscaled_embedding) + feat_s4)) # (b*v, 128, 128, 128)
            trans = conv5(upscaled_embedding)

        trans = trans.view(bs, self.num_img, 1, h, w).half()            
        # t_end_vggt8 = time.time()
        # if rank == 0:
        #     print("[DEBUG] Action (trans) calculated. Time Cost: {} minutes".format((t_end_vggt8 - t_start_vggt8) / 60.0)) 
            
        if not self.no_feat:
            # get wpt_local while testing
            if not self.training:
                wpt_local = self.get_wpt(
                    out={"trans": trans.clone().detach()}, vggt1_or_vggt2=False
                )
            # wpt_img = get_pt_loc_on_img(                      # (b, 1, 4, 2)
            #     wpt_local.unsqueeze(1), intrinsics, extrinsics
            # )
            wpt_img = self.renderer.get_pt_loc_on_img(
                wpt_local.unsqueeze(1), fix_cam=True, dyn_cam_info=None
            )
            wpt_img = wpt_img.reshape(bs * self.num_img, 2)

            # add noise to wpt image while training
            if self.training:
                wpt_img = add_uni_noi(
                    wpt_img, self.wpt_img_aug * self.img_size
                )
            wpt_img = torch.clamp(wpt_img, 0, self.img_size - 1)

            _wpt_img = wpt_img / self.img_patch_size
            _u = x
            assert (
                0 <= _wpt_img.min() and _wpt_img.max() <= x.shape[-1]
            ), print(_wpt_img, x.shape)

            _wpt_img = _wpt_img.unsqueeze(1)                                # (b*4, 1, 2)
            _feat = select_feat_from_hm(_wpt_img, _u)[0]
            _feat = _feat.view(bs, 1, -1) 

            feat.append(_feat)
            feat = torch.cat(feat, dim=-1)
            feat = feat.squeeze(1)

            # t_start_vggt9 = time.time()
            # features except rotation
            feat_ex_rot = self.feat_fc_ex_rot(feat)

            # batch normalized features for rotation
            feat_rot = self.feat_fc_init_bn(feat)
            feat_x = self.feat_fc_x(feat_rot)

            if self.training:
                rot_x = rot_x_y[..., 0].view(bs, 1)
            else:
                # sample with argmax
                rot_x = feat_x.argmax(dim=1, keepdim=True)

            rot_x_pe = self.feat_fc_pe(rot_x)
            feat_y = self.feat_fc_y(feat_rot + rot_x_pe)

            if self.training:
                rot_y = rot_x_y[..., 1].view(bs, 1)
            else:
                rot_y = feat_y.argmax(dim=1, keepdim=True)
            rot_y_pe = self.feat_fc_pe(rot_y)
            feat_z = self.feat_fc_z(feat_rot + rot_x_pe + rot_y_pe)
            # t_end_vggt9 = time.time()
            # if rank == 0:
            #     print("[DEBUG] Action (rot+others) calculated. Time Cost: {} minutes".format((t_end_vggt9 - t_start_vggt9) / 60.0)) 
            out = {
                "feat_ex_rot": feat_ex_rot.unsqueeze(1),
                "feat_x": feat_x.unsqueeze(1),
                "feat_y": feat_y.unsqueeze(1),
                "feat_z": feat_z.unsqueeze(1),
            }
        else:
            out = {}

        out.update({"trans": trans,})
        if self.no_feat:
            feature_1, feature_2, feature_3, feature_4 = self.get_matching_feature(match_input_dict)
            out.update({"feature_1": feature_1, "feature_2": feature_2, "feature_3": feature_3, "feature_4": feature_4})
                    # "depth": depth,
                    # "depth_conf": depth_conf
        # t_end_vggt10 = time.time()
        # if rank == 0:
        #     print("[DEBUG] End timing for RVT-2 multi-view transformer. Total Time Cost: {} minutes".format((t_end_vggt10 - t_end_vggt6) / 60.0))

        return out

    def get_wpt(self, out, vggt1_or_vggt2=False, intrinsics=None, extrinsics=None):
        """
        Estimate the q-values given output from vggt
        :param out: output from vggt

        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]  # 1
        q_trans = out["trans"].view(bs, nc, h * w)          

        if vggt1_or_vggt2:
            hm = F.softmax(q_trans, 2)    # add an additional temperature factor
            hm = hm.view(bs, nc, h, w)
            intrinsics_ = []
            extrinsics_ = []
            
            for i in range(bs):
                sample_intrinsics = torch.stack([K[i] for K in intrinsics])
                sample_extrinsics = torch.stack([E[i] for E in extrinsics])
                intrinsics_.append(sample_intrinsics)
                extrinsics_.append(sample_extrinsics)

            pred_wpt = [
                get_max_3d_frm_hm_cube( 
                    hm[i : i + 1], 
                    intrinsics_[i],    # (4, 3, 3)
                    extrinsics_[i]     # (4, 4, 4)
                )
                for i in range(bs)
            ]
        else:
            hm = F.softmax(q_trans, 2)    
            hm = hm.view(bs, nc, h, w)
            pred_wpt = [
                self.renderer.get_max_3d_frm_hm_cube(
                    hm[i : i + 1],
                    fix_cam=True,
                    dyn_cam_info=None,
                )
                for i in range(bs)
            ]    

        pred_wpt = torch.cat(pred_wpt, 0)
        pred_wpt = pred_wpt.squeeze(1)

        return pred_wpt


    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()