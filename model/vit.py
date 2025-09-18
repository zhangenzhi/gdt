import torch
from torch import nn
from typing import List, Dict
from functools import partial
from torch.utils.checkpoint import checkpoint

import timm
from timm.models.layers import DropPath
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
    """
    Generates positional embeddings from patch center coordinates and sizes.
    This is crucial for the model to understand the irregular patch layout.
    """
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

class SHFVisionTransformer(nn.Module):
    """
    A Vision Transformer adapted for the SHF dataloader.
    It takes a dictionary of serialized patches and their metadata as input.
    """
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
        super().__init__()
        
        # --- 1. Custom Embedding Layers ---
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = SpatioStructuralPosEmbed(embed_dim, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- 2. Core Transformer Backbone (from timm) ---
        timm_vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
        self.blocks = timm_vit.blocks
        self.norm = timm_vit.norm
        
        # --- 3. Classification Head ---
        self.head = timm_vit.head
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch_dict):
        # 1. Unpack the dictionary from the dataloader
        patches = batch_dict['patches'] # Shape: [B, N, C, P, P]
        sizes = batch_dict['sizes']       # Shape: [B, N]
        positions = batch_dict['positions'] # Shape: [B, N, 2]
        
        B, N, C, P, _ = patches.shape
        
        # 2. Flatten and embed patches
        x = patches.flatten(2)         # Shape: [B, N, C*P*P]
        x = self.patch_embed(x)        # Shape: [B, N, D_embed]

        # 3. Add the custom spatio-structural positional embedding
        x = x + self.pos_embed(positions, sizes)

        # 4. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [B, N+1, D_embed]
        
        # 5. Pass through Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # 6. Pass CLS token to classification head
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits
    
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
        in_channels=model_config.get('in_channels', 3),
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

