import torch
from torch import nn
from typing import List, Dict
from functools import partial
from torch.utils.checkpoint import checkpoint

import timm
from timm.layers import DropPath
import timm.models.vision_transformer
from timm.models.vision_transformer import VisionTransformer 

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding.
    Converts a 2D image into a 1D sequence of patch embeddings.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use a single Conv2d layer for efficient patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [B, C, H, W]
        # After proj: [B, D, H/P, W/P]
        # After flatten and transpose: [B, N, D] where N = (H*W)/P^2
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# class VisionTransformer(nn.Module):
#     """Standard Vision Transformer with a Transformer Encoder."""
#     def __init__(self, *, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         num_patches = (img_size // patch_size) ** 2

#         # Special tokens
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=dropout)

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, 
#             nhead=num_heads, 
#             dim_feedforward=int(embed_dim * mlp_ratio), 
#             dropout=dropout, 
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

#         # Classifier Head
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         x = self.transformer_encoder(x)
        
#         # Get the CLS token for classification
#         cls_output = self.norm(x[:, 0])
#         logits = self.head(cls_output)
        
#         # Return only logits to match standard classifier output
#         return logits

class RelativeAttention(nn.Module):
    def __init__(self, dim, num_heads, max_rel_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.max_rel_distance = max_rel_distance  # 假设最大距离范围
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * max_rel_distance - 1) ** 2, num_heads)
        )  # [2D distance_bucket, num_heads]

        # 坐标 index 到 bias_table 的映射表
        self.register_buffer("relative_index", self._build_relative_index(max_rel_distance))

    def _build_relative_index(self, max_dist):
        coords = torch.stack(torch.meshgrid(
            torch.arange(max_dist), torch.arange(max_dist), indexing='ij'
        ), dim=0)  # (2, max, max)

        coords_flat = coords.reshape(2, -1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        rel_coords += max_dist - 1  # shift to >=0
        rel_index = rel_coords[:, :, 0] * (2 * max_dist - 1) + rel_coords[:, :, 1]
        return rel_index  # (N, N)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # q/k/v: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        relative_bias = self.relative_bias_table[self.relative_index[:N, :N].reshape(-1)]
        relative_bias = relative_bias.reshape(N, N, -1).permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + relative_bias.unsqueeze(0)  # broadcast to (B, num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class RelativeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, max_rel_distance=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeAttention(dim, num_heads, max_rel_distance=max_rel_distance)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioStructuralPosEmbed(nn.Module):
    def __init__(self, embed_dim, img_size=224):
        super().__init__()
        self.img_size = float(img_size)
        
        # An MLP to project the geometric features [cx, cy, size] into the embedding space
        self.proj = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, positions, sizes):
        # Normalize features to a [0, 1] range
        # positions are (cx, cy)
        pos_norm = positions / self.img_size
        sizes_norm = sizes.unsqueeze(-1) / self.img_size
        
        # Concatenate features and project
        features = torch.cat([pos_norm, sizes_norm], dim=-1)
        pos_embed = self.proj(features)
        
        return pos_embed

class SHFVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 **kwargs):
        
        # --- 1. Call the Parent Constructor ---
        # This initializes all the standard ViT components like blocks, norm, head, etc.
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            **kwargs
        )
        
        # --- 2. Override Components for SHF Data Format ---
        
        # Override the patch_embed layer. Instead of a Conv2d, we need a Linear layer.
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # --- [CRITICAL FIX] ---
        # The parent class expects `pos_embed` to be an nn.Parameter or None.
        # We must not assign a module to it. Instead, we set it to None and
        # create a new attribute for our dynamic positional embedding module.
        self.pos_embed = None 
        self.dynamic_pos_embed = SpatioStructuralPosEmbed(embed_dim, img_size)
        
        # We still need a learnable positional embedding for the CLS token.
        # We create it as a separate parameter, matching the parent's convention.
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ----------------------
        
        # Re-apply weight initialization for the newly created layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_embed(self, x: torch.Tensor, positions: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        # 1. Generate dynamic positional embedding for patch tokens and add it.
        patch_pos_embed = self.dynamic_pos_embed(positions, sizes)
        x = x + patch_pos_embed

        # 2. Prepend the class token, now with its own dedicated positional embedding.
        # This follows the standard ViT practice (concat then apply dropout).
        cls_token_with_pos = self.cls_token + self.cls_pos_embed
        x = torch.cat([cls_token_with_pos.expand(x.shape[0], -1, -1), x], dim=1)
        
        return self.pos_drop(x)

    def forward_features(self, batch_dict: dict) -> torch.Tensor:
        # 1. Unpack data and embed patches
        patches = batch_dict['patches']
        x = self.patch_embed(patches.flatten(2))

        # 2. Add positional embedding (this now includes cls_token logic and dropout)
        x = self._pos_embed(x, batch_dict['positions'], batch_dict['sizes'])

        # 3. Apply patch dropout (inherited from parent)
        x = self.patch_drop(x)
        
        # 4. Apply pre-normalization (inherited from parent)
        x = self.norm_pre(x)
        
        # 5. Pass through Transformer blocks (inherited from parent)
        x = self.blocks(x)
        
        # 6. Apply final normalization (inherited from parent)
        x = self.norm(x)
        
        return x

    def forward(self, batch_dict: dict) -> torch.Tensor:
        # 1. Extract features using our custom logic that is now fully aligned with timm
        x = self.forward_features(batch_dict)
        
        # 2. Call the parent's head-forwarding logic
        # This correctly handles different global_pool settings ('token', 'avg', etc.)
        x = self.forward_head(x)
        
        return x
    
class MAEVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(MAEVisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, attn_mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def create_vit_model(config: Dict) -> VisionTransformer:
    """
    Factory function to create a VisionTransformer from a config dictionary.
    """
    model_config = config['model']
    model = VisionTransformer(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config.get('in_channels', 3),
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        num_classes=model_config['num_classes'],
    )
    return model


def create_timm_vit(config):  
    model_name = 'vit_base_patch16_224' 
    model_config = config['model']
    
    # 使用 timm.create_model 创建模型
    model = timm.create_model(
        model_name,
        pretrained=model_config['pretrained'],
        num_classes=model_config['num_classes'],
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config.get('in_channels', 3),
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
        weight_init = 'jax_nlhb',
        qkv_bias=True,
        # qk_norm = True,
        # init_values=1e-6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    return model

import torch
import torch.nn as nn
import timm
from timm.layers import RotaryEmbedding
from timm.models.vision_transformer import VisionTransformer, Block, Attention
from typing import Dict

# 确保已安装 timm
# pip install timm

# --- 步骤 1: 创建注入了 RoPE 的 Attention 模块 ---
# 我们继承 timm 的 Attention 类，只修改 forward 方法来应用 RoPE
class RopeAttention(Attention):
    """
    Attention module with Rotary Positional Embedding.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, rope=None, **kwargs):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)
        # 接收一个外部传入的 RoPE 实例
        self.rope = rope

    def forward(self, x):
        B, N, C = x.shape
        # self.qkv, self.scale, self.proj 都继承自父类 timm.models.vision_transformer.Attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # --- RoPE 注入点 ---
        # 只有在传入了 rope 实例时才应用
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        # --- 注入结束 ---

        x = self.forward_attention(q, k, v)
        x = self.proj(x)
        
        return x

class VisionTransformerWithRoPE(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,  # 使用timm的命名习惯 'in_chans'
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True, # 明确添加 qkv_bias
                 **kwargs):

        # --- 关键步骤 ---
        # 调用父类构造函数，并强制设置我们需要的架构选项
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            class_token=False,  # 必须禁用 class token
            pos_embed='none',   # 必须禁用 position embedding
            global_pool='avg',  # 必须设置为 avg pooling
            **kwargs            # 传递任何其他可能的参数
        )

        # 实例化 RoPE 模块
        head_dim = embed_dim // num_heads
        self.rope = RotaryEmbedding(dim=head_dim)

        # 重建 blocks，注入我们自定义的 RopeAttention
        # 注意: self.mlp_ratio, self.qkv_bias 等属性已由 super().__init__ 设置好
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_class=partial(RopeAttention, rope=self.rope),
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                drop=self.drop_rate,
                drop_path=self.dpr[i].item() if hasattr(self, 'dpr') and self.dpr is not None else 0.,
            )
            for i in range(depth)])

def create_rope_vit_model(config: Dict) -> VisionTransformerWithRoPE:
    """
    Factory function to create a VisionTransformer with RoPE from a config dictionary.
    """
    model_config = config['model']
    
    # 使用我们的 VisionTransformerWithRoPE 类
    model = VisionTransformerWithRoPE(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config.get('in_chans', 3), # 'in_channels' in user prompt, timm uses 'in_chans'
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        num_classes=model_config['num_classes'],
        qkv_bias=model_config.get('qkv_bias', True) # ViT-Base and others usually use bias
    )
    return model