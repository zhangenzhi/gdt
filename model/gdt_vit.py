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
# 模型创建函数
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
    num_patches_current_stage = (img_size // stages_config[0]['patch_size_in'])**2
    total_leaf_nodes = 0

    print("--- 正在根据配置计算总叶子结点数量 ---")
    for i, config in enumerate(stages_config):
        k_selected_ratio = config['k_selected_ratio']
        num_leaves_in_stage = math.ceil(num_patches_current_stage * (1.0 - k_selected_ratio))
        total_leaf_nodes += num_leaves_in_stage
        
        print(f"阶段 {i+1}: 输入 {num_patches_current_stage} 个 patch, 产生 {num_leaves_in_stage} 个叶子。")

        num_parents_for_next_stage = num_patches_current_stage - num_leaves_in_stage
        children_per_parent = (config['patch_size_in'] // config['patch_size_out'])**2
        num_patches_current_stage = num_parents_for_next_stage * children_per_parent

    # 最后剩下的 patch 也都是叶子结点
    total_leaf_nodes += num_patches_current_stage
    print(f"最后一个阶段后剩下 {num_patches_current_stage} 个叶子。")
    print(f"计算出的总叶子结点数量: {total_leaf_nodes}")
    print("---------------------------------------")
    
    TOTAL_LEAF_TOKENS = total_leaf_nodes # 使用之前精确计算的结果

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
        num_heads=12,
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
