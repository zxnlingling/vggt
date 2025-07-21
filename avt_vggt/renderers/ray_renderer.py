import torch
import numpy as np
import torch.nn as nn
import xformers.ops as xops
from einops import rearrange
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from utils.env_utils import IMAGE_SIZE, CAMERAS
from utils.mvt_utils import select_feat_from_hm, select_feat_from_hm_cache


# src: https://github.com/pytorch/benchmark/blob/main/torchbenchmark/models/llama/model.py#L28
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight.type_as(x)
    

class QK_Norm_SelfAttention(nn.Module):
    """
    Self-attention with optional Q-K normalization.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L68-L92
    """

    def __init__(
        self,
        dim,
        head_dim,
        qkv_bias=False,
        fc_bias=True,
        attn_dropout=0.0,
        fc_dropout=0.0,
        use_qk_norm=True,
    ):
        """
        Args:
            dim: Input dimension
            head_dim: Dimension of each attention head
            qkv_bias: Whether to use bias in QKV projection
            fc_bias: Whether to use bias in output projection
            attn_dropout: Dropout probability for attention weights
            fc_dropout: Dropout probability for output projection
            use_qk_norm: Whether to use Q-K normalization
        We use flash attention V2 for efficiency.
        """
        super().__init__()
        assert dim % head_dim == 0, f"Token dimension {dim} should be divisible by head dimension {head_dim}"
        
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.attn_dropout = attn_dropout
        self.use_qk_norm = use_qk_norm

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim, bias=fc_bias).half()
        self.attn_fc_dropout = nn.Dropout(fc_dropout).half()
        
        # Optional Q-K normalization
        if self.use_qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            attn_bias: Optional attention bias mask
            
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = (rearrange(t, "b l (nh dh) -> b l nh dh", dh=self.head_dim) for t in (q, k, v))
        
        # Apply qk normalization if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        x = xops.memory_efficient_attention(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
            attn_bias=attn_bias,
            p=self.attn_dropout if self.training else 0.0,
            op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
        )
        
        x = rearrange(x, "b l nh dh -> b l (nh dh)")
        x = self.attn_fc_dropout(self.fc(x))
        
        return x
    

class MLP(nn.Module):
    """
    Multi-Layer Perceptron block.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L49-L65
    """
    
    def __init__(
        self,
        dim,
        mlp_ratio=4,
        bias=False,
        dropout=0.0,
        activation=nn.GELU,
        mlp_dim=None,
    ):
        """
        Args:
            dim: Input dimension
            mlp_ratio: Multiplier for hidden dimension
            bias: Whether to use bias in linear layers
            dropout: Dropout probability
            activation: Activation function
            mlp_dim: Optional explicit hidden dimension (overrides mlp_ratio)
        """
        super().__init__()
        hidden_dim = mlp_dim if mlp_dim is not None else int(dim * mlp_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            activation(),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)
    

class QK_Norm_TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-normalization architecture.
    Reference: https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/vision_transformer.py#L95-L113
    """

    def __init__(
        self,
        dim,
        head_dim,
        ln_bias=False,
        attn_qkv_bias=False,
        attn_dropout=0.0,
        attn_fc_bias=False,
        attn_fc_dropout=0.0,
        mlp_ratio=4,
        mlp_bias=False,
        mlp_dropout=0.0,
        use_qk_norm=True,
        device="cuda"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=ln_bias)
        self.attn = QK_Norm_SelfAttention(
            dim=dim,
            head_dim=head_dim,
            qkv_bias=attn_qkv_bias,
            fc_bias=attn_fc_bias,
            attn_dropout=attn_dropout,
            fc_dropout=attn_fc_dropout,
            use_qk_norm=use_qk_norm,
        )

        self.norm2 = nn.LayerNorm(dim, bias=ln_bias)
        self.mlp = MLP(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=mlp_bias,
            dropout=mlp_dropout,
        )
        self.attn.to(device)
        self.norm1.to(device)
        self.mlp.to(device)
        self.norm2.to(device)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class RayRenderer():
    """
    Novel View Synthesis based on Pl¨ucker rays instead of known camera parameters
    Partially copied from official code for "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias"

    """
    def __init__(self, render_img_size, render_view_num, intrinsics_type, with_depth=True, device="cuda"):
        self.device = device
        self.with_depth = with_depth
        self.render_img_size = render_img_size
        self.render_view_num = render_view_num
        self.intrinsics_type = intrinsics_type
        self.predict_view = 'wrist'

        self.patch_size = 8
        self.d_head = 64
        self.d_model = 768

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()

        # Initialize transformer blocks
        self._init_transformer()

        self.image_tokenizer.to(device)
        self.target_pose_tokenizer.to(device)
        self.transformer_input_layernorm.to(device)
        self.image_token_decoder.to(device)

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = 9,
            patch_size = self.patch_size,
            d_model = self.d_model
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = 6,
            patch_size = self.patch_size,
            d_model = self.d_model
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.d_model, bias=False),
            nn.Linear(
                self.d_model,
                (self.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(self.init_weights)

    def _init_transformer(self):
        """Initialize transformer blocks"""
        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                self.d_model, self.d_head, use_qk_norm=True, device=self.device
            ) for _ in range(24)
        ]
        
        for block in self.transformer_blocks:
            block.apply(self.init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(self.d_model, bias=False)


    def _get_render_cameras(self, intrinsics):
        if self.render_view_num == 4:
            new_c2ws = [
                torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=torch.float32),     # front view
                torch.tensor([[0, 0, 1, 0], [1, 0, 0, -2], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),    # left view
                torch.tensor([[0, 0, -1, 0], [-1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),   # right view
                torch.tensor([[0, -1, 0, 0], [0, 0, 1, 2], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),    # top view
            ]
        elif self.render_view_num == 3:
            new_c2ws = [
                torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=torch.float32),     # front view
                torch.tensor([[0, 0, -1, 0], [-1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),   # right view
                torch.tensor([[0, -1, 0, 0], [0, 0, 1, 2], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),    # top view
            ]
        elif self.render_view_num == 2:
            new_c2ws = [
                torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=torch.float32),     # front view
                torch.tensor([[0, -1, 0, 0], [0, 0, 1, 2], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.float32),    # top view
            ]
        else: # render only the front view
            new_c2ws = [torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=torch.float32),]

        all_c2ws = torch.stack(new_c2ws, dim=0) # [num_views, 4, 4]
        self.all_c2ws = all_c2ws.unsqueeze(0)   # (1, num_views, 4, 4)

        # Construct intrinsic parameters
        all_fxfycxcy_list = []
        for i in range(self.render_view_num):
            if self.intrinsics_type == "rvt":       # use intrinsics of rvt
                img_sizes_w = [2, 2]
                img_h = self.render_img_size
                img_w = int(img_h / self.original_image_h * self.original_image_w)
                fx = img_h / img_sizes_w[0]
                fy = img_w / img_sizes_w[1]
                cx = img_h / 2
                cy = img_w / 2
            else:   # use intrinsics of input cameras
                current_K = intrinsics[i]           # the i-th view: (1, 3, 3)
                fx = current_K[0, 0].item()
                fy = current_K[1, 1].item()
                cx = current_K[0, 2].item()
                cy = current_K[1, 2].item()
                if self.intrinsics_type == "adaptive": # use intrinsics scale to input cameras
                    img_h = self.render_img_size
                    img_w = int(img_h / self.original_image_h * self.original_image_w)
                    fx *= img_h / self.original_image_h
                    fy *= img_w / self.original_image_w 
                    cx *= img_h / self.original_image_h 
                    cy *= img_w / self.original_image_w 
                fxfycxcy = torch.tensor([fx, fy, cx, cy], device=self.device, dtype=torch.float32)
                all_fxfycxcy_list.append(fxfycxcy)
        all_fxfycxcy_views = torch.stack(all_fxfycxcy_list, dim=0)
        self.all_fxfycxcy = all_fxfycxcy_views.unsqueeze(0)  # [1, num_views, 4]

        # Construct intrinsic matrices [num_views, 3, 3]
        base_intrinsics_list = []
        for i in range(self.render_view_num):
            fxfycxcy = all_fxfycxcy_views[i]  # (4,)
            K = torch.tensor([
                [fxfycxcy[0], 0, fxfycxcy[2]],
                [0, fxfycxcy[1], fxfycxcy[3]],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device)
            base_intrinsics_list.append(K)
        base_intrinsics = torch.stack(base_intrinsics_list, dim=0)  # (num_views, 3, 3)
        self.all_intrinsics = base_intrinsics.unsqueeze(0)          # (1, num_views, 3, 3)
        return self
    
    def free_mem(self):
        del self.image_tokenizer
        del self.target_pose_tokenizer
        del self.image_token_decoder
        torch.cuda.empty_cache()  

    
    def init_weights(module, std=0.02):
        """Initialize weights for linear and embedding layers.
        
        Args:
            module: Module to initialize
            std: Standard deviation for normal initialization
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(self.init_weights)
        return tokenizer

    def preprocess_views(self, input_images, rlbench_intrinsics, rlbench_extrinsics):
        resize_h = self.render_img_size
        square_crop = False

        intrinsics = []
        w2cs = [] 

        for i, camera_view in enumerate(['front', 'left_shoulder', 'right_shoulder', 'wrist']):

            fx = rlbench_intrinsics[i, 0, 0]
            fy = rlbench_intrinsics[i, 1, 1]
            cx = rlbench_intrinsics[i, 0, 2]
            cy = rlbench_intrinsics[i, 1, 2]
            fxfycxcy = torch.tensor([fx, fy, cx, cy], device=self.device)
            w2c = rlbench_extrinsics[i]

            self.original_image_h, self.original_image_w = input_images.shape[2], input_images.shape[3]
            resize_h = self.render_img_size
            resize_w = int(resize_h / self.original_image_h * self.original_image_w)

            resize_ratio_x = resize_w / self.original_image_w
            resize_ratio_y = resize_h / self.original_image_h
            scale_factors = torch.tensor(
                [resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y],
                device=self.device,
                dtype=fxfycxcy.dtype
            )
            fxfycxcy = fxfycxcy * scale_factors

            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h

            intrinsics.append(fxfycxcy)
            w2cs.append(w2c)

        intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)
        w2cs = torch.stack(w2cs, dim=0)
        c2ws = torch.inverse(w2cs).unsqueeze(0)

        # manually specify the novel camera views
        self.cameras = self._get_render_cameras(rlbench_intrinsics)
        return intrinsics, c2ws


    def _check_device(self, input, input_name):
        assert str(input.device) == str(self.device), (
            f"Input {input_name} (device {input.device}) should be on the same device as the renderer ({self.device})")
    

    @torch.no_grad()
    def get_pt_loc_on_img(self, pt, fix_cam=False, dyn_cam_info=None):
        """
        Project 3D points onto image planes of the novel views.
        :param pt: (bs, np, 3) 3D points in world coordinates
        :return: (bs, np, num_views, 2) 2D pixel locations on each novel view
        """
        assert len(pt.shape) == 3 and pt.shape[-1] == 3
        assert fix_cam, "Not supported with ray renderer"
        assert dyn_cam_info is None, "Not supported with ray renderer"

        bs, np, _ = pt.shape
        num_views = self.all_c2ws.shape[1]

        inv_poses = self.all_c2ws                                       # [bs, num_views, 4, 4]
        intrinsics = self.all_intrinsics                                # [bs, num_views, 3, 3]

        pcs_px = []
        for i in range(bs):
            # Project points from world to camera frame
            pc = pt[i]  # [np, 3]
            pc_h = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1)   # [np, 4]
            pc_cam_h = torch.einsum('vxy,nx->nvx', inv_poses[i], pc_h)  # [num_views, np, 4]
            pc_cam = pc_cam_h[..., :3]                                  # [num_views, np, 3]
            z = pc_cam[..., 2:]                                         # [num_views, np, 1]
            xy = pc_cam[..., :2] / z                                    # [num_views, np, 2]
            uv = torch.einsum('vxy, nvx -> nvx', intrinsics[i][:, :, :2], xy.unsqueeze(-1)).squeeze(-1)  # [num_views, np, 2]
            pcs_px.append(uv)

        pcs_px = torch.stack(pcs_px, dim=0)                             # [bs, num_views, np, 2]
        pcs_px = pcs_px.permute(0, 2, 1, 3)                             # [bs, np, num_views, 2]

        return pcs_px
    
    @torch.no_grad()
    def get_feat_frm_hm_cube(self, hm, fix_cam=False, dyn_cam_info=None):
        """
        :param hm: torch.Tensor of (1, num_img, h, w)
        :return: tupe of ((num_img, h^3, 1), (h^3, 3))
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == self.render_view_num
        assert self.render_img_size == h and h == w
        assert fix_cam, "Not supported with ray renderer"
        assert dyn_cam_info is None, "Not supported with ray renderer"

        if self._pts is None:
            res = self.render_img_size
            pts = torch.linspace(-1 + (1 / res), 1 - (1 / res), res, device=hm.device)
            pts = torch.cartesian_prod(pts, pts, pts)
            self._pts = pts

        pts_hm = []

        # if self._fix_cam
        if self._fix_pts_cam is None:
            # (np, nc, 2)
            pts_img = self.get_pt_loc_on_img(self._pts.unsqueeze(0),
                                             fix_cam=True).squeeze(0)
            # pts_img = pts_img.permute((1, 0, 2))
            # (nc, np, bs)
            fix_pts_hm, pts_cam, pts_cam_wei = select_feat_from_hm(
                pts_img.transpose(0, 1), hm.transpose(0, 1)[0 : len(self.cameras)]
            )
            self._fix_pts_img = pts_img
            self._fix_pts_cam = pts_cam
            self._fix_pts_cam_wei = pts_cam_wei
        else:
            pts_cam = self._fix_pts_cam
            pts_cam_wei = self._fix_pts_cam_wei
            fix_pts_hm = select_feat_from_hm_cache(
                pts_cam, hm.transpose(0, 1)[0 : len(self.cameras)], pts_cam_wei
            )
        pts_hm.append(fix_pts_hm)

        #if not dyn_cam_info is None:
        # TODO(Valts): implement
        pts_hm = torch.cat(pts_hm, 0)
        return pts_hm, self._pts
    
    @torch.no_grad()
    def get_max_3d_frm_hm_cube(self, hm, fix_cam=False, dyn_cam_info=None,
                               topk=1, non_max_sup=False,
                               non_max_sup_dist=0.02):
        """
        given set of heat maps, return the 3d location of the point with the
            largest score, assumes the points are in a cube [-1, 1]. This function
            should be used  along with the render. 
        :param hm: (1, nc, h, w)
        :return: (1, topk, 3)
        """
        assert fix_cam, "Not supported with ray renderer"
        assert dyn_cam_info is None, "Not supported with ray renderer"

        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == len(self.cameras)
        assert self.render_img_size == h and h == w

        pts_hm, pts = self.get_feat_frm_hm_cube(hm, fix_cam, dyn_cam_info)
        # (bs, np, nc)
        pts_hm = pts_hm.permute(2, 1, 0)
        # (bs, np)
        pts_hm = torch.mean(pts_hm, -1)
        if non_max_sup and topk > 1:
            _pts = pts.clone()
            pts = []
            pts_hm = torch.squeeze(pts_hm, 0)
            for i in range(topk):
                ind_max_pts = torch.argmax(pts_hm, -1)
                sel_pts = _pts[ind_max_pts]
                pts.append(sel_pts)
                dist = torch.sqrt(torch.sum((_pts - sel_pts) ** 2, -1))
                pts_hm[dist < non_max_sup_dist] = -1
            pts = torch.stack(pts, 0).unsqueeze(0)
        else:
            # (bs, topk)
            ind_max_pts = torch.topk(pts_hm, topk)[1]
            # (bs, topk, 3)
            pts = pts[ind_max_pts]
        return pts
    
    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, original_h=None, original_w=None):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """
        assert c2w.dim() == 4 and c2w.shape[-2:] == (4, 4), f"Expected c2w to be (b, v, 4, 4), but got {c2w.shape}"
        assert fxfycxcy.dim() == 3 and fxfycxcy.shape[-1] == 4, f"Expected fxfycxcy to be (b, v, 4), but got {fxfycxcy.shape}"

        b, v = c2w.shape[:2]
        c2w = c2w.reshape(b * v, 4, 4).to(self.device)

        fx, fy, cx, cy = fxfycxcy.unbind(dim=-1)  # (b, v)
        if original_h is None or original_w is None:
            # estimate the original height/width from the intrinsic matrix)
            h_orig = int(2 * cy.max().item())
            w_orig = int(2 * cx.max().item())
        else:
            h_orig = original_h 
            w_orig = original_w  
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = torch.stack([fx, fy, cx, cy], dim=-1).reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), 
                              indexing="ij")
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d).to(self.device)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    @torch.no_grad()
    def process_data(self, input_images, intrinsics, extrinsics, compute_rays = True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.
                
        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        all_camera_views = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        target_view_index = [all_camera_views.index(self.predict_view)]
        input_view_indices = [i for i, view in enumerate(all_camera_views) if view != self.predict_view]

        input_dict, target_dict = {}, {}
        input_intrinsics, input_c2ws = self.preprocess_views(input_images, intrinsics, extrinsics)

        input_dict["image"] = input_images[:, input_view_indices, ...]
        input_dict["c2w"] = input_c2ws[:, input_view_indices, ...]              # (1, 4, 4, 4)
        input_dict["fxfycxcy"] = input_intrinsics[:, input_view_indices, :]   # (1, 4, 4)
        input_dict["index"] = input_view_indices
        input_dict["scene_name"] = "rlbench"
        height, width = input_images.shape[3], input_images.shape[4]
        input_dict["image_h_w"] = (height, width)

        target_dict["image"] = input_images[:, target_view_index, ...]
        target_dict["c2w"] = input_c2ws[:, target_view_index, :, :]
        target_dict["fxfycxcy"] = input_intrinsics[:, target_view_index, :]
        target_dict["index"] = target_view_index
        target_dict["scene_name"] = input_dict["scene_name"]
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)
        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]
                ray_o, ray_d = self.compute_rays(c2w, fxfycxcy, image_height, image_width, 
                                                 original_h=IMAGE_SIZE, original_w=IMAGE_SIZE)
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        """
        input_dict:
            input_dict["image"]     ([1, 3, 3, 128, 128])
            input_dict["c2w"]   ([[[[ 1.1685e-07, -1.0000e+00, -6.0318e-07,  8.3243e-07],
                                    [-4.2262e-01, -5.6357e-07,  9.0631e-01, -8.6143e-01],
                                    [-9.0631e-01,  1.3126e-07, -4.2262e-01,  1.8913e+00],
                                    [ 2.3566e-08,  2.1384e-15, -1.7638e-08,  1.0000e+00]],

                                    [[ 1.7365e-01,  9.8481e-01, -1.7526e-07, -1.6657e-01],
                                    [ 8.9254e-01, -1.5738e-01,  4.2262e-01, -6.4911e-01],
                                    [ 4.1620e-01, -7.3387e-02, -9.0631e-01,  1.8820e+00],
                                    [ 6.4110e-09,  8.7455e-10, -7.5974e-09,  1.0000e+00]],

                                    [[-1.7365e-01,  9.8481e-01, -8.4069e-08,  1.6657e-01],
                                    [ 8.9254e-01,  1.5738e-01,  4.2262e-01, -6.4911e-01],
                                    [ 4.1620e-01,  7.3387e-02, -9.0631e-01,  1.8820e+00],
                                    [-5.4351e-09,  3.9609e-10,  1.0915e-08,  1.0000e+00]]]], device='cuda:2')
            input_dict["fxfycxcy"] ([[[-7501.9873, -7502.4453,  2730.5000,  2730.6667],
                                    [-7501.9873, -7502.4453,  2730.5000,  2730.6667],
                                    [-7501.9873, -7502.4453,  2730.5000,  2730.6667]]], device='cuda:2')
            input_dict["index"]     [0, 1, 2]
            input_dict["scene_name"]: rlbench
            input_dict["image_h_w"] (128, 128)
            input_dict["ray_o"]     (1, 1, 3, 128, 128)
            input_dict["ray_d"]     (1, 1, 3, 128, 128)
        """

        return input_dict, target_dict
    
    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
        
    def pass_layers(self, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(self.transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in self.transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = self.transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens

    @torch.no_grad()
    def render_batch(self, pc, input_images, intrinsics, extrinsics):
        """
        Render novel views from the model.
        
        """
    
        input, target = self.process_data(input_images, intrinsics, extrinsics, compute_rays = True)
        
        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)                       # [b*v_input, n_patches, d]
        _, n_patches, d = input_img_tokens.size()                                   
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)     # [b, v_input*n_patches, d]

        # Compute rays for rendering novel views
        rendering_ray_o, rendering_ray_d = self.compute_rays(
            fxfycxcy=self.all_fxfycxcy, c2w=self.all_c2ws, h=h, w=w, original_h=IMAGE_SIZE, original_w=IMAGE_SIZE
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o, 
            ray_d=rendering_ray_d
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond)                   # [b * v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)       # [b, v_target * n_patches, d]

        rendered_images = []
        rendered_depths = []

        for i in range(num_views):

            # Extract the i-th view's pose tokens
            cur_target_pose_tokens = target_pose_tokens[:, i * n_patches : (i + 1) * n_patches]

            # Repeat input image tokens for the current view
            repeated_input_img_tokens = input_img_tokens.repeat_interleave(1, dim=0)        # [b, v_input*n_patches, d]

            # Concatenate input and target pose tokens                                      [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) 
            cur_concat_input_tokens = self.transformer_input_layernorm(cur_concat_input_tokens)

            # Pass through transformer and split output tokens
            transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)
            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_patches, n_patches], dim=1
            )

            # Decode image tokens and reshape to image format
            height, width = input.image_h_w
            cur_image = self.image_token_decoder(pred_target_image_tokens)
            cur_image = rearrange(
                cur_image, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v = 1,
                h = height // self.patch_size, 
                w = width // self.patch_size, 
                p1 = self.patch_size, 
                p2 = self.patch_size, 
                c = 3
            )

            rendered_images.append(cur_image)

            if self.with_depth:
                # depth extraction
                pass

        rendered_images = torch.cat(rendered_images, dim=1)
        # rendered_depths = torch.cat(rendered_depths, dim=1)                 # [b, v, 1, h, w]

        return rendered_images, rendered_depths




    def __call__(self, pc, feat, input_images, intrinsics, extrinsics):

        # check device
        assert str(pc.device) == str(self.device), (
            f"Input {pc} (device {pc.device}) should be on the same device as the renderer ({self.device})")
        assert str(feat.device) == str(self.device), (
            f"Input {feat} (device {feat.device}) should be on the same device as the renderer ({self.device})")
        assert str(input_images.device) == str(self.device), (
            f"Input {input_images} (device {input_images.device}) should be on the same device as the renderer ({self.device})")
        
        rendered_images, rendered_depths = self.render_batch(pc, input_images, intrinsics, extrinsics)

        # normalize output
        # _, h, w = rendered_depths.shape
        # depth_0 = rendered_depths == -1
        # depth_sum = torch.sum(rendered_depths, (1, 2)) + torch.sum(depth_0, (1, 2))
        # depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
        # rendered_depths -= depth_mean.unsqueeze(-1).unsqueeze(-1)
        # rendered_depths[depth_0] = -1

        # if self.with_depth:
        #     img_out = torch.cat([rendered_images, rendered_depths], dim=2)  # [b, v, 3+1, h, w]
        # else:
        img_out = rendered_images

        return img_out
 

    

    