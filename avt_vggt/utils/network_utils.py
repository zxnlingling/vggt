import math
import torch
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat


# constants
LRELU_SLOPE = 0.02

# tools
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def act_layer(act):
    if act == "relu":
        return nn.ReLU()
    elif act == "lrelu":
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == "elu":
        return nn.ELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "prelu":
        return nn.PReLU()
    else:
        raise ValueError("%s not recognized." % act)
    
def norm_layer2d(norm, channels):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    elif norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == "group":
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)

def norm_layer1d(norm, num_channels):
    if norm == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    elif norm == "layer":
        return nn.LayerNorm(num_channels)
    elif norm == "group":
        return nn.GroupNorm(4, num_channels, affine=True)
    else:
        raise ValueError("%s not recognized." % norm)
    
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


# used in self-attention layers of MVT
def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)
    
class Attention(nn.Module):  # is all you need. Living up to its name.
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64,
                 dropout=0.0, use_fast=False):

        super().__init__()
        self.use_fast = use_fast
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout_p = dropout
        # dropout left in use_fast for backward compatibility
        self.dropout = nn.Dropout(self.dropout_p)

        self.avail_xf = False

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        if self.use_fast:
            # use PyTorch Flash Attention
            out = F.scaled_dot_product_attention(
                query=q, 
                key=k, 
                value=v, 
                attn_mask=mask,  
                dropout_p=self.dropout_p if self.training else 0.0, 
            )
            # # using py2 if available
            # dropout_p = self.dropout_p if self.training else 0.0
            # # using xf if available
            # if self.avail_xf:
            #     out = xops.memory_efficient_attention(
            #         query=q, key=k, value=v, p=dropout_p
            #     )
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
            if exists(mask):
                mask = rearrange(mask, "b ... -> b (...)")
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, "b j -> (b h) () j", h=h)
                sim.masked_fill_(~mask, max_neg_value)
            # attention
            attn = sim.softmax(dim=-1)
            # dropout
            attn = self.dropout(attn)
            out = einsum("b i j, b j d -> b i d", attn, v)

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        return out


# used in self.patchify() to obtain the original rgb features
class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super().__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

        self.activation = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        else:
            self.norm = None
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x
    

# used in proprio and language preprocessing, self.fc_bef_attn() and self.fc_aft_attn()
class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        if activation is None:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.linear.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.linear.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            nn.init.zeros_(self.linear.bias)
        else:
            raise ValueError()

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x
    

# used in self.feat_fc_pe() to get rot features (feat_x, feat_y, feat_z)
# based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FixedPositionalEncoding(nn.Module):
    def __init__(self, feat_per_dim: int, feat_scale_factor: int):
        super().__init__()
        self.feat_scale_factor = feat_scale_factor
        # shape [1, feat_per_dim // 2]
        div_term = torch.exp(
            torch.arange(0, feat_per_dim, 2) * (-math.log(10000.0) /
                                                feat_per_dim)
        ).unsqueeze(0)
        self.register_buffer("div_term", div_term)

    def forward(self, x):
        """
        :param x: Tensor, shape [batch_size, input_dim]
        :return: Tensor, shape [batch_size, input_dim * feat_per_dim]
        """
        assert len(x.shape) == 2
        batch_size, input_dim = x.shape
        x = x.view(-1, 1)
        x = torch.cat((
            torch.sin(self.feat_scale_factor * x * self.div_term),
            torch.cos(self.feat_scale_factor * x * self.div_term)), dim=1)
        x = x.view(batch_size, -1)
        return x
    

# used in self.up0 to get trans prediction
class ConvexUpSample(nn.Module):
    """
    Learned convex upsampling with optional target size control
    """

    def __init__(
        self, in_dim, out_dim, up_ratio, up_kernel=3, mask_scale=0.1, with_bn=False, 
    ):
        """

        :param in_dim:
        :param out_dim:
        :param up_ratio:
        :param up_kernel:
        :param mask_scale:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        self.with_bn = with_bn

        assert (self.up_kernel % 2) == 1

        if with_bn:
            self.net_out_bn1 = nn.BatchNorm2d(2 * in_dim)
            self.net_out_bn2 = nn.BatchNorm2d(2 * in_dim)

        self.net_out = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, out_dim, 3, padding=1),
        )

        mask_dim = (self.up_ratio**2) * (self.up_kernel**2)
        self.net_mask = nn.Sequential(
            nn.Conv2d(in_dim, 2 * in_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * in_dim, mask_dim, 1, padding=0),
        )

    def forward(self, x):
        """

        :param x: (bs, in_dim, h, w)
        :return: (bs, out_dim, h*up_ratio, w*up_ratio)
        """

        bs, c, h, w = x.shape
        assert c == self.in_dim, c

        # low resolution output
        if self.with_bn:
            out_low = self.net_out[0](x)
            out_low = self.net_out_bn1(out_low)
            out_low = self.net_out[1](out_low)
            out_low = self.net_out[2](out_low)
            out_low = self.net_out_bn2(out_low)
            out_low = self.net_out[3](out_low)
            out_low = self.net_out[4](out_low)
        else:
            out_low = self.net_out(x)

        mask = self.mask_scale * self.net_mask(x)
        mask = mask.view(bs, 1, self.up_kernel**2, self.up_ratio, self.up_ratio, h, w)
        mask = torch.softmax(mask, dim=2)

        out = F.unfold(
            out_low,
            kernel_size=[self.up_kernel, self.up_kernel],
            padding=self.up_kernel // 2,
        )
        out = out.view(bs, self.out_dim, self.up_kernel**2, 1, 1, h, w)

        out = torch.sum(out * mask, dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(bs, self.out_dim, h * self.up_ratio, w * self.up_ratio)

        return out

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# used in self.fusion() to get VGGT rgb features from the raw vggt_out
# from SAM-E
class Fusion_up(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_sizes=3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,):
        super(Fusion_up, self).__init__()
        # Convolutional layer to change channels 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides)
        
        if activation is None:
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("linear")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "tanh":
            nn.init.xavier_uniform_(
                self.conv2d.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "lrelu":
            nn.init.kaiming_uniform_(
                self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.conv2d.bias)
        elif activation == "relu":
            nn.init.kaiming_uniform_(self.conv2d.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv2d.bias)
        else:
            raise ValueError()

    def forward(self, x):
        # Apply convolutional layer
        x = self.conv2d(x)
        # Upsample to change spatial dimensions from to 16x16
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        return x
    

# LoRA tools
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
        
class LoRAQkv(nn.Module):
    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv

class LoRAConv2d(nn.Module):

    # Takes a torch.nn.Conv2d, LoRA config and builds an adapter
    def __init__(self, base_module: nn.Conv2d, lora_config: dict) -> None:
        super(LoRAConv2d, self).__init__()

        assert isinstance(base_module, nn.Conv2d), 'Invalid type! The base module should be of type `torch.nn.Conv2d`.'

        self.base_module = base_module
        for parameter in self.base_module.parameters():
            parameter.requires_grad = False
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        # Creating trainable parameters & registering buffers
        self.register_buffer(
            'alpha',
            torch.tensor(
                data=(lora_config['alpha'],),
                dtype=self.base_module.weight.dtype,
                device=self.base_module.weight.device
            )
        )

        self.register_buffer(
            'rank',
            torch.tensor(
                data=(lora_config['rank'],),
                dtype=torch.int,
                device=self.base_module.weight.device
            )
        )

        assert 'rank_for' in lora_config.keys(), 'Missing `rank_for` in `lora_config`! Please provide `rank_for` with valid values: "kernel", "channels".'
        assert lora_config['rank_for'] in ['kernel', 'channels'], 'Invalid `rank_for` value! Please pick from the valid values: "kernel", "channels".'
        self.rank_for = lora_config['rank_for']
        if self.rank_for == 'kernel':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, kH, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, lora_config['rank'], kW),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
        elif lora_config['rank_for'] == 'channels':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(kH, kW, out_channels, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(kH, kW, lora_config['rank'], in_channels),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B
        self.reset_trainable_parameters()
        self.adapter_enabled = False  # Controls the inferencing, "base" or "base + adapter"

    # Resetting/initializing trainable parameters: delta_weight_A, delta_weight_B
    def reset_trainable_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.delta_weight_B, a=5 ** 0.5)

    # Sets inference state, forward pass happens through base_module + adapter
    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    # Sets inference state, forward pass happens through base_module
    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    # Creates a new instance of type torch.nn.Conv2d with merged weight of base module and LoRA adapter
    def get_merged_module(self) -> nn.Conv2d:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
        elif self.rank_for == 'channels':
            effective_weight = self.base_module.weight + (
                    (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1))
        effective_bias = self.base_module.bias

        merged_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kH, kW),
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            bias=effective_bias is not None,
            padding_mode=self.base_module.padding_mode
        )
        merged_module.weight.data = effective_weight
        if effective_bias is not None:
            merged_module.bias.data = effective_bias
        return merged_module

    # State dependent (adapter_enabled) forward propagation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_enabled:
            if self.rank_for == 'kernel':
                effective_weight = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
            elif self.rank_for == 'channels':
                effective_weight = self.base_module.weight + (
                        (self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1))
        else:
            effective_weight = self.base_module.weight
        effective_bias = self.base_module.bias

        return nn.functional.conv2d(
            x,
            weight=effective_weight,
            bias=effective_bias,
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups
        )

    # Modify the representation string to include LoRA parameters
    def __repr__(self) -> str:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            adapter_repr_string = f'Adapter(kH={kH}, rank={self.rank.item()}, kW={kW})'
        elif self.rank_for == 'channels':
            adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_features={out_channels})'

        repr_string = f'LoRAConv2d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        return repr_string