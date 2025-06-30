import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple
import numpy as np

# --- 图像处理和可视化所需的库 ---
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# 为了方便，我们使用现成的 TransformerEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ======================================================================
# 基础模块 
# ======================================================================
class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_coord_val: int = 512):
        super().__init__()
        if embed_dim % 2 != 0: raise ValueError(f"embed_dim must be even, got {embed_dim}")
        half_dim = embed_dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim))
        self.register_buffer('div_term', div_term)
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        pos_x, pos_y = coords[..., 0].unsqueeze(-1), coords[..., 1].unsqueeze(-1)
        pe_x_sin, pe_x_cos = torch.sin(pos_x * self.div_term), torch.cos(pos_x * self.div_term)
        pe_y_sin, pe_y_cos = torch.sin(pos_y * self.div_term), torch.cos(pos_y * self.div_term)
        pe_x = torch.cat([pe_x_sin, pe_x_cos], dim=-1)
        pe_y = torch.cat([pe_y_sin, pe_y_cos], dim=-1)
        return torch.cat([pe_x, pe_y], dim=-1)

class GumbelTopKSelector(nn.Module):
    def __init__(self, input_dim: int, num_items: int):
        super().__init__()
        self.scorer = nn.Linear(input_dim, num_items)
    def forward(self, features: torch.Tensor, k: int, temperature: float = 1.0):
        logits = self.scorer(features)
        gumbel_scores = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        _, top_k_indices = torch.topk(gumbel_scores, k, dim=-1, sorted=False)
        hard_mask = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, top_k_indices, 1.0)
        ste_mask = (hard_mask - gumbel_scores).detach() + gumbel_scores
        return ste_mask, top_k_indices

# ======================================================================
# 编码器部分 (Hierarchical ViT Encoder)
# ======================================================================
class HierarchicalStage(nn.Module):
    def __init__(self, *, embed_dim: int, depth: int, num_heads: int, patch_size_in: int, patch_size_out: int, k_selected_ratio: float, max_seq_len: int, in_channels: int, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_size_in, self.patch_size_out, self.k_selected_ratio = patch_size_in, patch_size_out, k_selected_ratio
        self.patch_embed = nn.Linear(in_channels * patch_size_in * patch_size_in, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_relative = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))
        self.pos_encoder_absolute = SinusoidalPositionalEncoder(embed_dim)
        self.pos_drop = nn.Dropout(p=dropout)
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim*mlp_ratio), dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.selector = GumbelTopKSelector(input_dim=embed_dim, num_items=max_seq_len)
        self.unfolder = nn.Unfold(kernel_size=patch_size_out, stride=patch_size_out)
        
    def forward(self, raw_patches_in: torch.Tensor, coords_in: torch.Tensor) -> Tuple:
        B, N_in, _ = raw_patches_in.shape
        C = raw_patches_in.shape[-1] // (self.patch_size_in ** 2)
        k_selected = int(N_in * self.k_selected_ratio)
        tokens_in = self.patch_embed(raw_patches_in)
        abs_pe = self.pos_encoder_absolute(coords_in)
        tokens_with_abs_pe = tokens_in + abs_pe
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, tokens_with_abs_pe), dim=1)
        x = x + self.pos_embed_relative[:, :(N_in + 1)]
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        cls_token_output = x[:, 0]
        ste_mask, top_k_indices = self.selector(cls_token_output, k=k_selected)
        
        differentiable_raw_patches = raw_patches_in * ste_mask.unsqueeze(-1)
        indices_expanded_patches_raw = top_k_indices.unsqueeze(-1).expand(-1, -1, raw_patches_in.shape[-1])
        selected_raw_patches = torch.gather(differentiable_raw_patches, 1, indices_expanded_patches_raw)
        
        selected_patches_img = selected_raw_patches.view(B * k_selected, C, self.patch_size_in, self.patch_size_in)
        raw_patches_out = self.unfolder(selected_patches_img).transpose(1, 2).reshape(B, -1, C * self.patch_size_out * self.patch_size_out)
        
        selected_coords = torch.gather(coords_in, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, 2))
        p_out = self.patch_size_out
        sub_coords_y, sub_coords_x = torch.meshgrid(torch.arange(0, self.patch_size_in, p_out), torch.arange(0, self.patch_size_in, p_out), indexing='ij')
        sub_coords = torch.stack([sub_coords_x, sub_coords_y], dim=-1).float().to(x.device).view(1, -1, 2)
        coords_out = selected_coords.unsqueeze(2) + sub_coords
        coords_out = coords_out.reshape(B, -1, 2)
        
        return raw_patches_out, coords_out, top_k_indices

class HierarchicalViTEncoder(nn.Module):
    def __init__(self, *, img_size: int, stages_config: List[Dict], in_channels: int = 3, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.stages_config = stages_config
        self.initial_patch_size = stages_config[0]['patch_size_in']
        self.initial_unfolder = nn.Unfold(kernel_size=self.initial_patch_size, stride=self.initial_patch_size)
        gy, gx = torch.meshgrid(torch.arange(0, img_size, self.initial_patch_size), torch.arange(0, img_size, self.initial_patch_size), indexing='ij')
        initial_coords = torch.stack([gx, gy], dim=-1).float().reshape(-1, 2)
        self.register_buffer('initial_coords', initial_coords)
        self.stages = nn.ModuleList([HierarchicalStage(embed_dim=embed_dim, num_heads=num_heads, in_channels=in_channels, **config) for config in stages_config])
    
    def forward(self, x: torch.Tensor) -> List[Dict]:
        B = x.shape[0]
        current_raw_patches = self.initial_unfolder(x).transpose(1, 2)
        current_coords = self.initial_coords.unsqueeze(0).expand(B, -1, -1)
        all_leaf_nodes_data = []

        for stage in self.stages:
            N_in = current_raw_patches.shape[1]
            raw_patches_out, coords_out, selected_indices_local = stage(current_raw_patches, current_coords)
            unselected_mask = torch.ones(B, N_in, dtype=torch.bool, device=x.device)
            unselected_mask.scatter_(1, selected_indices_local, False)
            unselected_patches = current_raw_patches[unselected_mask].reshape(B, -1, current_raw_patches.shape[-1])
            if unselected_patches.numel() > 0:
                all_leaf_nodes_data.append({'patches': unselected_patches, 'size': stage.patch_size_in})
            current_raw_patches = raw_patches_out
            current_coords = coords_out
        
        all_leaf_nodes_data.append({'patches': current_raw_patches, 'size': self.stages[-1].patch_size_out})
        return all_leaf_nodes_data

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
# 顶层模型 (AdaptiveFocusViT) - 组装所有部件
# ======================================================================
class AdaptiveFocusViT(nn.Module):
    def __init__(self, encoder: HierarchicalViTEncoder, classifier: DownstreamViTClassifier, target_leaf_size: int, embed_dim: int, in_channels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.target_leaf_size = target_leaf_size
        self.in_channels = in_channels
        self.leaf_embedder = nn.Linear(in_channels * target_leaf_size * target_leaf_size, embed_dim)
        
    def forward(self, img: torch.Tensor):
        B = img.shape[0]
        
        # 1. 从编码器获取所有叶子结点
        leaf_nodes_data = self.encoder(img)
        
        # 2. 将所有叶子 patch 缩放到统一尺寸并嵌入
        resized_leaf_tokens = []
        for leaf_group in leaf_nodes_data:
            raw_patches = leaf_group['patches'] # Shape: [B, Num_patches, C*S*S]
            if raw_patches.numel() == 0: continue
            
            size_in = leaf_group['size']
            num_patches_in_group = raw_patches.shape[1]
            
            patches_as_imgs = raw_patches.view(B * num_patches_in_group, self.in_channels, size_in, size_in)
            
            # 使用 F.interpolate 进行可微分的缩放
            resized_patches_imgs = F.interpolate(patches_as_imgs, size=(self.target_leaf_size, self.target_leaf_size), mode='bilinear', align_corners=False)
            
            resized_flat_patches = resized_patches_imgs.view(B, num_patches_in_group, -1)
            
            tokens = self.leaf_embedder(resized_flat_patches)
            resized_leaf_tokens.append(tokens)

        # 3. 拼接所有叶子 token 成为一个序列
        final_sequence = torch.cat(resized_leaf_tokens, dim=1)
        
        # 4. 送入下游分类器
        import pdb; pdb.set_trace()
        logits = self.classifier(final_sequence)
        
        return logits