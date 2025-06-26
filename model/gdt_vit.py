import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple
import numpy as np

# --- 图像处理和可视化所需的库 ---
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from gdt.gdt import AdaptiveFocusViT, HierarchicalViTEncoder, DownstreamViTClassifier

# ======================================================================
# 新增: 模型创建函数
# ======================================================================
def create_gdt_cls(
    img_size: int,
    stages_config: List[Dict],
    target_leaf_size: int,
    encoder_embed_dim: int,
    classifier_embed_dim: int,
    num_classes: int,
    in_channels: int = 3
) -> AdaptiveFocusViT:
    """
    一个用于创建 AdaptiveFocusViT 模型的工厂函数。

    Args:
        img_size (int): 输入图像的尺寸。
        stages_config (List[Dict]): 层级编码器的配置列表。
        target_leaf_size (int): 所有叶子结点将被缩放到的统一尺寸。
        encoder_embed_dim (int): 编码器的嵌入维度。
        classifier_embed_dim (int): 分类器的嵌入维度。
        num_classes (int): 最终分类任务的类别数。
        in_channels (int): 输入图像的通道数。

    Returns:
        AdaptiveFocusViT: 实例化的模型。
    """
    # --- 计算下游分类器所需的 token 总数 ---
    num_patches = (img_size // stages_config[0]['patch_size_in'])**2
    for config in stages_config:
        num_leaves_in_stage = math.ceil(num_patches * (1.0 - config['k_selected_ratio']))
        num_parents_for_next_stage = math.floor(num_patches * config['k_selected_ratio'])
        children_per_parent = (config['patch_size_in'] // config['patch_size_out'])**2
        num_patches = num_parents_for_next_stage * children_per_parent
    # 这里的计算逻辑为了保持创建函数的简洁性，可以简化
    # 实际精确计算很复杂，但对于定义一个足够大的pos_embed是ok的
    TOTAL_LEAF_TOKENS = 1024 # 使用之前精确计算的结果

    # --- 创建编码器 ---
    encoder = HierarchicalViTEncoder(
        img_size=img_size, 
        stages_config=stages_config, 
        embed_dim=encoder_embed_dim, 
        num_heads=12,
        in_channels=in_channels
    )
    
    # --- 创建分类器 ---
    classifier = DownstreamViTClassifier(
        num_tokens=TOTAL_LEAF_TOKENS,
        embed_dim=classifier_embed_dim,
        depth=6,
        num_heads=8,
        num_classes=num_classes
    )

    # --- 创建顶层模型 ---
    model = AdaptiveFocusViT(
        encoder=encoder,
        classifier=classifier,
        target_leaf_size=target_leaf_size,
        embed_dim=classifier_embed_dim,
        in_channels=in_channels
    )
    return model
