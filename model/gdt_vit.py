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

from gdt.gdt import HierarchicalViTEncoder


# ======================================================================
# 下游分类器部分 (Downstream Classifier) - 负责分类
# ======================================================================
class DownstreamViTClassifier(nn.Module):
    def __init__(self, *, num_tokens: int, embed_dim: int, depth: int, num_heads: int, num_classes: int, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio), dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x[:, 0])
    
# ======================================================================
# 顶层模型
# ======================================================================
class AdaptiveFocusViT(nn.Module):
    def __init__(self, encoder: HierarchicalViTEncoder, classifier: DownstreamViTClassifier, target_leaf_size: int, embed_dim: int, in_channels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.target_leaf_size = target_leaf_size
        self.in_channels = in_channels
        self.leaf_embedder = nn.Linear(in_channels * target_leaf_size * target_leaf_size, embed_dim)
        
    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        B = img.shape[0]
        leaf_nodes_data = self.encoder(img)
        resized_leaf_tokens = []
        for leaf_group in leaf_nodes_data:
            raw_patches = leaf_group['patches']
            if raw_patches.numel() == 0: continue
            size_in = leaf_group['size']
            num_patches_in_group = raw_patches.shape[1]
            patches_as_imgs = raw_patches.view(B * num_patches_in_group, self.in_channels, size_in, size_in)
            resized_patches_imgs = F.interpolate(patches_as_imgs, size=(self.target_leaf_size, self.target_leaf_size), mode='bilinear', align_corners=False)
            resized_flat_patches = resized_patches_imgs.view(B, num_patches_in_group, -1)
            tokens = self.leaf_embedder(resized_flat_patches)
            resized_leaf_tokens.append(tokens)
        final_sequence = torch.cat(resized_leaf_tokens, dim=1)
        logits = self.classifier(final_sequence)
        return logits, leaf_nodes_data

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


def visualize_evaluation_focus(original_image: Image.Image, leaf_nodes_data: List[Dict], target_leaf_size: int, img_size: int, output_filename: str):
    """在评估时可视化模型的注意力焦点。"""
    
    # 准备画布和颜色
    resized_orig = original_image.resize((img_size, img_size), Image.Resampling.LANCZOS)
    leaf_map_canvas = Image.new("RGB", (img_size, img_size), "black")
    resampled_map_canvas = Image.new("RGB", (img_size, img_size), "black")
    
    # 定义从大 patch (蓝色) 到小 patch (红色) 的颜色梯度
    colors = [(0, 0, 255, 150), (0, 128, 255, 150), (0, 255, 255, 150), (255, 255, 0, 150), (255, 0, 0, 150)]
    patch_sizes = sorted(list(set(item['size'] for item in leaf_nodes_data)), reverse=True)
    color_map = {size: colors[i % len(colors)] for i, size in enumerate(patch_sizes)}

    print("开始生成评估可视化图像...")

    for group in leaf_nodes_data:
        # 我们只可视化 batch 中的第一个样本
        patches_tensor = group['patches'][0]
        coords_tensor = group['coords'][0]
        patch_size = group['size']
        color = color_map.get(patch_size, (255, 255, 255, 150))

        for i in range(patches_tensor.shape[0]):
            coord = coords_tensor[i].cpu().numpy()
            x, y = int(coord[0]), int(coord[1])
            
            # --- Part 2: 绘制带颜色的叶子结点图 ---
            patch_tensor_chw = patches_tensor[i].view(3, patch_size, patch_size)
            patch_img = transforms.ToPILImage()(patch_tensor_chw)
            
            color_overlay = Image.new("RGBA", patch_img.size, color)
            colored_patch = Image.alpha_composite(patch_img.convert("RGBA"), color_overlay)
            leaf_map_canvas.paste(colored_patch, (x, y))

            # --- Part 3: 绘制统一采样后还原的图 ---
            patch_as_img_tensor = patches_tensor[i].view(1, 3, patch_size, patch_size)
            resampled_patch_tensor = F.interpolate(patch_as_img_tensor, size=(target_leaf_size, target_leaf_size), mode='bilinear', align_corners=False)
            restored_patch_tensor = F.interpolate(resampled_patch_tensor, size=(patch_size, patch_size), mode='nearest')
            restored_patch_img = transforms.ToPILImage()(restored_patch_tensor.squeeze(0))
            resampled_map_canvas.paste(restored_patch_img, (x, y))
            
    # --- 拼接最终图像 ---
    final_img = Image.new('RGB', (img_size * 3 + 20, img_size), (255, 255, 255))
    final_img.paste(resized_orig, (0, 0))
    final_img.paste(leaf_map_canvas, (img_size + 10, 0))
    final_img.paste(resampled_map_canvas, (img_size * 2 + 20, 0))
    
    draw = ImageDraw.Draw(final_img)
    try: font = ImageFont.truetype("arial.ttf", 12)
    except IOError: font = ImageFont.load_default()
    draw.text((10, 10), "1. Original Image", fill="black", font=font)
    draw.text((img_size + 20, 10), "2. Leaf Node Map (Colored by Size)", fill="black", font=font)
    draw.text((img_size * 2 + 30, 10), "3. Resampled & Restored Map", fill="black", font=font)

    final_img.save(output_filename)
    print(f"评估可视化结果已保存到文件: {output_filename}")