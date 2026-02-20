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