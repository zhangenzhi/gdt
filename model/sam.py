"""
SAM-B (ViT-Base) 模型工厂，支持动态调整Patch Size以适应超高分辨率输入，并包含分割解码器。
根据图像尺寸自动适配输入通道数 (1k/8k: 1通道, 32k: 3通道)。
"""
import logging
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import (
    PatchEmbed,
    Mlp,
    DropPath,
    LayerNorm2d,
    LayerScale,
    Format,
    resample_abs_pos_embed_nhwc,
    to_2tuple,
    use_fused_attn,
)
from torch.jit import Final

# --- 辅助函数和字典 (来自您的timm代码) ---

_logger = logging.getLogger(__name__)

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size, dtype=torch.float32)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, dtype=torch.float32)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def get_decomposed_rel_pos_bias(
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn_bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn_bias.reshape(-1, q_h * q_w, k_h * k_w)


# --- 核心模型模块 (来自您的timm代码) ---

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            use_rel_pos: bool = False,
            input_size: Optional[Tuple[int, int]] = None,
            rope: Optional[nn.Module] = None,
            device=None, # 保留以允许传递，但仅用于nn.Parameter
            dtype=None,
    ):
        super().__init__()
        # 用于nn.Parameter的参数字典
        param_dd = {'device': device, 'dtype': dtype}
        # 用于不支持device/dtype的层的参数字典
        layer_dd = {} # 通常为空

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # 标准层不支持 device/dtype
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **layer_dd)
        self.q_norm = norm_layer(self.head_dim, **layer_dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **layer_dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, **layer_dd)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert rope is None
            assert (
                    input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # nn.Parameter 支持 device/dtype
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim, **param_dd))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim, **param_dd))
        self.rope = rope

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rel_pos:
            attn_bias = get_decomposed_rel_pos_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        else:
            attn_bias = None
            if self.rope is not None:
                raise NotImplementedError("ROPE is not implemented in this simplified example.")

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.view(B, self.num_heads, N, -1).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, H, W, -1)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, hw: Tuple[int, int], pad_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    Hp, Wp = pad_hw if pad_hw is not None else hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    x = x[:, :H, :W, :].contiguous()
    return x


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            use_rel_pos: bool = False,
            window_size: int = 0,
            input_size=None,
            rope=None,
            device=None, # 保留以允许传递，但仅用于nn.Parameter
            dtype=None,
    ):
        super().__init__()
        # 用于nn.Parameter的参数字典
        param_dd = {'device': device, 'dtype': dtype}
        # 用于不支持device/dtype的层的参数字典
        layer_dd = {} # 通常为空

        self.window_size = window_size
        self.norm1 = norm_layer(dim, **layer_dd)
        # 注意：Attention内部已经处理了哪些层接收dd
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            rope=rope,
            device=device, # 传递给Attention，它会内部处理
            dtype=dtype,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, **layer_dd) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, **layer_dd)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            # **layer_dd, # <-- 修正：Mlp不支持device/dtype
        )
        self.ls2 = LayerScale(dim, init_values=init_values, **layer_dd) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, _ = x.shape
        shortcut = x
        x = self.norm1(x)

        pad_hw: Optional[Tuple[int, int]] = None
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        x = self.drop_path1(self.ls1(self.attn(x)))

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, (H, W), pad_hw)

        x = shortcut + x

        # MLP is faster for N, L, C tensor
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(self.ls2(x))
        x = shortcut + x

        return x


# --- *** 新增：分割解码器 *** ---
class SimpleSegmentationDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, upsample_factor=16):
        """
        一个简单的分割解码器，用于将ViT特征图上采样到原始分辨率。

        Args:
            in_channels (int): 输入特征图的通道数 (来自ViT neck)。
            out_channels (int): 输出分割图的通道数 (例如，1用于二元分割)。
            upsample_factor (int): 从特征图到原始图像尺寸的总上采样倍数。
                                   等于 patch_size。
        """
        super().__init__()
        # 计算需要多少个2倍上采样层
        num_upsample_layers = int(torch.log2(torch.tensor(float(upsample_factor))).item())

        layers = []
        current_channels = in_channels
        for i in range(num_upsample_layers):
            # 逐步减少通道数，最后一层除外，避免通道数过少
            out_ch = current_channels // 2 if current_channels > 32 and i < num_upsample_layers -1 else current_channels
            # 确保通道数至少为16
            out_ch = max(16, out_ch)

            layers.append(
                nn.ConvTranspose2d(current_channels, out_ch, kernel_size=2, stride=2)
            )
            # 添加BatchNorm和ReLU以提高稳定性
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_ch

        # 最后的1x1卷积，将通道数降为目标输出通道数
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [B, C_neck, H_grid, W_grid]
        return self.decoder(x)

# --- *** 修改后的 VisionTransformerSAM *** ---
class VisionTransformerSAM(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            in_chans: int = 3,
            num_classes: int = 1, # 修改：现在控制解码器输出通道
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            pre_norm: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            embed_layer: Type[nn.Module] = partial(PatchEmbed, output_fmt=Format.NHWC, strict_img_size=False),
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
            act_layer: Optional[Type[nn.Module]] = nn.GELU,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            use_rope: bool = False,
            window_size: int = 14,
            global_attn_indexes: Tuple[int, ...] = (),
            neck_chans: int = 256,
            device=None, # 保留以允许传递，但仅用于nn.Parameter
            dtype=None,
    ):
        super().__init__()
        # 用于nn.Parameter的参数字典
        param_dd = {'device': device, 'dtype': dtype}
        # 用于不支持device/dtype的层的参数字典
        layer_dd = {} # 通常为空

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes # 现在是分割类别数
        self.embed_dim = embed_dim
        self.grad_checkpointing = False
        self.patch_size = patch_size # 保存patch_size以供解码器使用

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        grid_size = self.patch_embed.grid_size

        if use_abs_pos:
            # SAM uses a 64x64 pos embedding, which is resampled dynamically.
            # nn.Parameter 支持 device/dtype
            self.pos_embed = nn.Parameter(torch.zeros(1, 64, 64, embed_dim, **param_dd))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        # This is always False for SAM
        self.norm_pre = norm_layer(embed_dim, **layer_dd) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            # Block内部已经处理了哪些层接收dd
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=grid_size,
                rope=None, # ROPE is not used in SAM-B
                device=device, # 传递给Block，它会内部处理
                dtype=dtype,
            )
            for i in range(depth)])

        # SAM Neck
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                neck_chans,
                kernel_size=1,
                bias=False,
                **layer_dd,
            ),
            LayerNorm2d(neck_chans, **layer_dd),
            nn.Conv2d(
                neck_chans,
                neck_chans,
                kernel_size=3,
                padding=1,
                bias=False,
                **layer_dd,
            ),
            LayerNorm2d(neck_chans, **layer_dd),
        )

        # --- *** 新增：添加解码器 *** ---
        # patch_size 就是总的下采样倍数
        self.decoder = SimpleSegmentationDecoder(
            in_channels=neck_chans,
            out_channels=num_classes,
            upsample_factor=self.patch_size # 使用保存的patch_size
        )


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # Dynamically resize abs pos embedding
            x = x + resample_abs_pos_embed_nhwc(self.pos_embed, x.shape[1:3])
        x = self.pos_drop(x)
        x = self.norm_pre(x)

        # x shape is (B, H_grid, W_grid, C_embed)
        x = self.blocks(x)

        # x shape is (B, H_grid, W_grid, C_embed), permute to (B, C_embed, H_grid, W_grid) for Neck
        x = self.neck(x.permute(0, 3, 1, 2))
        # x shape is (B, C_neck, H_grid, W_grid)
        return x

    def forward(self, x):
        # --- *** 修改：通过解码器生成分割图 *** ---
        features = self.forward_features(x)
        segmentation_logits = self.decoder(features)
        # 确保输出尺寸与输入完全一致 (如果解码器设计精确，可能不需要插值)
        # 但为了鲁棒性，保留插值步骤
        if segmentation_logits.shape[2:] != x.shape[2:]:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        return segmentation_logits # 返回logits [B, num_classes, H, W]


# --- Checkpoint Loading Helpers (来自您的timm代码) ---

def checkpoint_filter_fn(
        state_dict,
        model,
):
    """ Remap SAM checkpoints -> timm """
    sam_checkpoint = 'image_encoder.patch_embed.proj.weight' in state_dict
    out_dict = {}
    is_segmentation_model = hasattr(model, 'decoder') # 检查是否有decoder

    for k, v in state_dict.items():
        if k.startswith('image_encoder.'):
            k = k[14:]
            k = k.replace('mlp.lin', 'mlp.fc')

            # 如果是分割模型，跳过原始checkpoint中的neck和head层
            # 修正: 应该保留neck层权重
            if is_segmentation_model and k.startswith('head.'):
                 _logger.debug(f"Skipping loading key '{k}' for segmentation model.")
                 continue

        elif sam_checkpoint: # 如果是原始SAM checkpoint，跳过非encoder部分
             # 不再需要特殊处理neck层，因为上面已经保留
            if k.startswith('head.'): # 跳过原始SAM的head层
                continue
            # 跳过其他非encoder部分 (例如 prompt_encoder, mask_decoder)
            if not k.startswith('patch_embed.') and \
               not k.startswith('pos_embed') and \
               not k.startswith('blocks.') and \
               not k.startswith('neck.'):
                continue

        # 将过滤后的键值对添加到新字典
        out_dict[k] = v

    return out_dict


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 0, 'input_size': (3, 1024, 1024), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head.fc',
        **kwargs
    }

default_cfgs = {
    'samvit_base_patch16.sa1b': _cfg(
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        hf_hub_id='timm/',
    ),
}


# --- *** 新的、灵活的模型工厂函数 *** ---

def create_sam_b_variant(
    img_size: int,
    # in_chans: int = 3, # 移除参数，内部根据img_size决定
    target_patch_grid_size: int = 64,
    pretrained: bool = True,
    num_classes: int = 1 # 添加参数以控制分割输出通道
) -> VisionTransformerSAM:
    """
    创建一个SAM-B（ViT-Base）变体模型，自动调整patch_size和in_chans以匹配目标网格大小和图像尺寸。
    包含一个简单的分割解码器。

    Args:
        img_size (int): 您的输入图像尺寸 (1024, 8192, 32768).
        target_patch_grid_size (int): 您希望Transformer处理的目标序列长度 (e.g., 64)。
        pretrained (bool): 是否加载预训练的SAM编码器权重。
        num_classes (int): 分割输出的通道数 (e.g., 1 for binary segmentation)。

    Returns:
        VisionTransformerSAM: 一个配置好并可选择性加载了权重的模型。
    """

    # --- *** 新增：根据img_size确定in_chans *** ---
    if img_size == 1024 or img_size == 8192:
        in_chans = 1
    elif img_size == 32768:
        in_chans = 3
    else:
        raise ValueError(f"不支持的图像尺寸: {img_size}。请使用 1024, 8192, 或 32768。")

    # 1. 自动计算新的patch size
    new_patch_size = img_size // target_patch_grid_size
    if img_size % target_patch_grid_size != 0:
        _logger.warning(
            f"Image size ({img_size}) is not perfectly divisible by grid size ({target_patch_grid_size}). "
            f"Resulting patch size will be {new_patch_size}, with some cropping."
        )

    _logger.info(
        f"创建SAM-B分割变体: img_size={img_size}, in_chans={in_chans}, target_grid={target_patch_grid_size}x{target_patch_grid_size} "
        f"-> 计算出的 patch_size={new_patch_size}x{new_patch_size}, 输出通道={num_classes}"
    )

    # 2. 定义SAM-B的架构参数
    model_args = dict(
        patch_size=new_patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        global_attn_indexes=[2, 5, 8, 11], # SAM-B的全局注意力层
        window_size=14,
        use_rel_pos=True,
        img_size=img_size,
        in_chans=in_chans, # 使用确定的通道数
        neck_chans=256,
        num_classes=num_classes, # 传递给模型以设置解码器输出
    )

    # 3. 从头创建模型
    model = VisionTransformerSAM(**model_args)

    # 4. 如果需要，加载并调整预训练权重
    if pretrained:
        _logger.info("正在加载预训练的SAM-B (sam_vit_b_01ec64.pth) 编码器权重...")
        cfg = default_cfgs['samvit_base_patch16.sa1b']

        # 从URL加载原始的状态字典
        state_dict = load_state_dict_from_url(cfg['url'], map_location='cpu')

        # 过滤掉不需要的键 (主要是原始SAM的解码器部分，如果存在的话)
        filtered_state_dict = checkpoint_filter_fn(state_dict, model)

        # --- **核心逻辑: 调整Patch Embedding权重** ---
        orig_patch_embed_key = 'patch_embed.proj.weight'
        # 检查过滤后的字典中是否还有这个键
        if orig_patch_embed_key in filtered_state_dict:
            orig_patch_embed_weights = filtered_state_dict[orig_patch_embed_key]
            orig_patch_size = 16 # SAM-B的原始patch size

            if new_patch_size != orig_patch_size:
                _logger.warning(
                    f"Patch size (new={new_patch_size}) 与 "
                    f"checkpoint (old={orig_patch_size}) 不匹配。"
                    f"正在使用双三次插值调整 patch_embed 权重。"
                )
                new_patch_embed_weights = F.interpolate(
                    orig_patch_embed_weights,
                    size=(new_patch_size, new_patch_size),
                    mode='bicubic',
                    align_corners=False,
                )
                filtered_state_dict[orig_patch_embed_key] = new_patch_embed_weights

            # 调整输入通道数 (如果需要)
            # 这里的in_chans已经是根据img_size确定的值
            if in_chans != 3:
                _logger.warning(
                    f"Input channels (new={in_chans}) 与 "
                    f"checkpoint (old=3) 不匹配。"
                    f"将对 patch_embed 权重的输入通道取平均值。"
                )
                # (embed_dim, 3, H_patch, W_patch) -> (embed_dim, 1, H_patch, W_patch)
                # 我们取3个通道的平均值
                new_weights = filtered_state_dict[orig_patch_embed_key].mean(dim=1, keepdim=True)
                if in_chans > 1: # 仅当目标通道数大于1时复制 (虽然在这里不会发生)
                    new_weights = new_weights.repeat(1, in_chans, 1, 1)
                filtered_state_dict[orig_patch_embed_key] = new_weights
            # 如果in_chans是3，则不需要做任何事
        else:
            _logger.warning(f"无法在预训练权重中找到 '{orig_patch_embed_key}'。跳过Patch Embedding调整。")


        # 加载位置编码 (Positional Embedding)
        # 我们的模型 pos_embed 尺寸是固定的 [1, 64, 64, 768]
        # 如果checkpoint中的尺寸不同，发出警告，但加载时strict=False会跳过它
        if 'pos_embed' in filtered_state_dict and filtered_state_dict['pos_embed'].shape != model.pos_embed.shape:
             _logger.warning(
                f"Positional embedding (new={model.pos_embed.shape}) 与 "
                f"checkpoint (old={filtered_state_dict['pos_embed'].shape}) 不匹配。"
                f"模型将在forward时自动插值。"
             )
             # strict=False 会处理这个问题，无需手动删除
             # del filtered_state_dict['pos_embed']

        # 加载状态字典
        # strict=False 允许加载部分匹配的权重 (例如，跳过解码器层)
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        _logger.info(f"预训练权重加载完成。")
        if missing_keys:
            # 过滤掉与解码器相关的缺失键，这些是预期中的
            missing_decoder_keys = [k for k in missing_keys if k.startswith('decoder.')]
            other_missing_keys = [k for k in missing_keys if not k.startswith('decoder.')]
            if missing_decoder_keys:
                _logger.info(f"缺失的权重 (预期中的解码器层): {len(missing_decoder_keys)} keys like '{missing_decoder_keys[0]}', ...")
            if other_missing_keys:
                 _logger.warning(f"意外缺失的权重: {other_missing_keys}")
        if unexpected_keys:
            _logger.warning(f"意外加载的权重 (可能来自原始SAM的非编码器部分): {unexpected_keys}")

    return model

# --- *** 本地测试 *** ---
if __name__ == '__main__':
    # 设置一个简单的日志记录器
    logging.basicConfig(level=logging.INFO)

    # 1. 1k 图像 -> 64x64 grid -> 16x16 patch, in_chans=1
    print("\n--- 测试 1k 图像 (1024x1024, 1通道) ---")
    model_1k = create_sam_b_variant(
        img_size=1024,
        target_patch_grid_size=64,
        # in_chans=1, # 移除，函数内部自动设置
        pretrained=True,
        num_classes=1 # 输出单通道分割图
    )
    dummy_1k = torch.randn(1, 1, 1024, 1024) # 输入通道改为1
    with torch.no_grad():
        output_1k = model_1k(dummy_1k)
    print(f"1k Model (patch: {model_1k.patch_embed.patch_size[0]}, in_chans: {model_1k.patch_embed.proj.in_channels}): "
          f"{dummy_1k.shape} -> {output_1k.shape}")
    assert output_1k.shape == (1, 1, 1024, 1024)
    print("1k 测试通过。")

    # 2. 8k 图像 -> 64x64 grid -> 128x128 patch, in_chans=1
    print("\n--- 测试 8k 图像 (8192x8192, 1通道) ---")
    model_8k = create_sam_b_variant(
        img_size=8192,
        target_patch_grid_size=64,
        # in_chans=1, # 移除
        pretrained=True,
        num_classes=1
    )
    dummy_8k = torch.randn(1, 1, 8192, 8192) # 输入通道改为1
    with torch.no_grad():
        output_8k = model_8k(dummy_8k)
    print(f"8k Model (patch: {model_8k.patch_embed.patch_size[0]}, in_chans: {model_8k.patch_embed.proj.in_channels}): "
          f"{dummy_8k.shape} -> {output_8k.shape}")
    assert output_8k.shape == (1, 1, 8192, 8192)
    print("8k 测试通过。")

    # 3. 32k 图像 -> 64x64 grid -> 512x512 patch, in_chans=3
    print("\n--- 测试 32k 图像 (32768x32768, 3通道) ---")
    model_32k = create_sam_b_variant(
        img_size=32768,
        target_patch_grid_size=64,
        # in_chans=3, # 移除
        pretrained=True,
        num_classes=1
    )
    dummy_32k = torch.randn(1, 3, 32768, 32768) # 输入通道保持3
    with torch.no_grad():
        output_32k = model_32k(dummy_32k)
    print(f"32k Model (patch: {model_32k.patch_embed.patch_size[0]}, in_chans: {model_32k.patch_embed.proj.in_channels}): "
          f"{dummy_32k.shape} -> {output_32k.shape}")
    assert output_32k.shape == (1, 1, 32768, 32768)
    print("32k 测试通过。")

