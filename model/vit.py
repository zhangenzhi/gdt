import torch
from torch import nn
from typing import List, Dict
from functools import partial
from torch.utils.checkpoint import checkpoint

import timm
from timm.models.layers import DropPath
import timm.models.vision_transformer

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
#         num_patches = self.patch_embed.num_patches

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

# import transformer_engine.pytorch as te
# from transformer_engine.common import recipe

# class PatchEmbedding(nn.Module):
#     """
#     Image to Patch Embedding.
#     Converts a 2D image into a 1D sequence of patch embeddings.
#     """
#     def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) ** 2
        
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x

# class VisionTransformerTE(nn.Module):
#     """
#     Vision Transformer optimized with NVIDIA Transformer Engine for FP8 training.
#     """
#     def __init__(self, *, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout=0.1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_classes = num_classes # Store original number of classes
        
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=dropout)

#         self.transformer_layers = nn.ModuleList([
#             te.TransformerLayer(
#                 hidden_size=embed_dim,
#                 ffn_hidden_size=int(embed_dim * mlp_ratio),
#                 num_attention_heads=num_heads,
#                 hidden_dropout=dropout,
#                 attention_dropout=dropout,
#                 self_attn_mask_type="no_mask", 
#                 activation="gelu"
#             ) for _ in range(depth)
#         ])

#         padded_num_classes = (num_classes + 15) & -16
        
#         self.norm = te.LayerNorm(embed_dim)
#         self.head = te.Linear(embed_dim, padded_num_classes, bias=True)

#         self.apply(self._init_weights)
        
#         # CORRECTED: Use a try-except block for robust hardware checking.
#         # This makes the model compatible with torch.compile on newer TE versions
#         # while maintaining compatibility with older versions.
#         try:
#             self.fp8_enabled, reason = te.is_fp8_available()
#             if not self.fp8_enabled:
#                 print(f"Warning: FP8 training is not available. Reason: {reason}")
#         except AttributeError:
#             print("Warning: 'transformer_engine.is_fp8_available()' not found.")
#             print("This may be due to an older version of TE. FP8 will be enabled by default.")
#             print("Note: Using torch.compile with this version may cause errors.")
#             self.fp8_enabled = True


#         self.fp8_recipe = recipe.DelayedScaling(
#             margin=0, 
#             interval=1, 
#             fp8_format=recipe.Format.HYBRID, 
#             amax_history_len=16, 
#             amax_compute_algo="max"
#         )

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         # Use the pre-checked boolean flag for the `enabled` argument.
#         with te.fp8_autocast(enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe):
#             B = x.shape[0]
#             x = self.patch_embed(x)

#             cls_tokens = self.cls_token.expand(B, -1, -1)
#             x = torch.cat((cls_tokens, x), dim=1)
#             x = x + self.pos_embed
#             x = self.pos_drop(x)

#             B, N, D = x.shape
#             pad_len = (16 - (N % 16)) % 16
#             if pad_len > 0:
#                 padding = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
#                 x = torch.cat([x, padding], dim=1)

#             for layer in self.transformer_layers:
#                 x = layer(x)
            
#             cls_output = self.norm(x[:, 0])
            
#             logits_padded = self.head(cls_output)
            
#             logits = logits_padded[:, :self.num_classes]
        
#         return logits
    
# class Block(nn.Module):
#     """
#     带有LayerScale的Transformer块 (Pre-Norm结构)。
#     """
#     def __init__(self, dim, num_heads, mlp_ratio=4., dropout=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         # 注意: PyTorch的MultiheadAttention默认dropout在softmax之后，这里我们保持一致
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(dropout)
#         )

#         # LayerScale参数
#         self.gamma_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
#         self.gamma_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

#     def forward(self, x):
#         # 注意力块 (Pre-Norm)
#         normed_x = self.norm1(x)
#         attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        
#         # 应用LayerScale并添加残差连接
#         if self.gamma_1 is not None:
#             x = x + self.gamma_1 * attn_output
#         else:
#             x = x + attn_output

#         # MLP块 (Pre-Norm)
#         normed_x = self.norm2(x)
#         mlp_output = self.mlp(normed_x)

#         # 应用LayerScale并添加残差连接
#         if self.gamma_2 is not None:
#             x = x + self.gamma_2 * mlp_output
#         else:
#             x = x + mlp_output
            
#         return x

# class VisionTransformer(nn.Module):
#     """
#     标准的视觉Transformer。
#     新增了 'use_checkpointing' 和 'layer_scale_init_value' 参数。
#     """
#     def __init__(self, *, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, 
#                  num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout=0.1, 
#                  use_checkpointing=False, layer_scale_init_value=1e-6):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
#         num_patches = (img_size // patch_size) ** 2

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=dropout)

#         # *** 修改: 使用自定义的Block模块列表替换nn.TransformerEncoder ***
#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 dropout=dropout,
#                 layer_scale_init_value=layer_scale_init_value
#             )
#             for _ in range(depth)])
#         # 为了与激活检查点兼容，我们将ModuleList包装在nn.Sequential中
#         self.transformer_encoder = nn.Sequential(*self.blocks)

#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_classes)
        
#         self.use_checkpointing = use_checkpointing
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None: nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward_features(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
        
#         if self.use_checkpointing and self.training:
#             x = checkpoint(self.transformer_encoder, x, use_reentrant=False)
#         else:
#             x = self.transformer_encoder(x)
            
#         return x

#     def forward(self, x):
#         x = self.forward_features(x)
#         cls_output = self.norm(x[:, 0])
#         logits = self.head(cls_output)
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

    def forward_features(self, x):
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
        # droppath=model_config.get('droppath', 0.0)        
        # layer_scale_init_value=float(model_config['layer_scale_init_value'])
    )
    return model


def create_timm_vit(config):  
    # 查找匹配的ViT模型名称，最常见的是 'vit_base_patch16_224'
    # 'base' 通常意味着 depth=12, embed_dim=768, num_heads=12
    model_name = 'vit_base_patch16_224' 
    model_config = config['model']
    
    print(f"正在基于 '{model_name}' 创建模型，并使用自定义参数进行覆盖。")
    
    # # 使用 timm.create_model 创建模型
    # model = timm.create_model(
    #     model_name,
    #     pretrained=model_config['pretrained'],
    #     num_classes=model_config['num_classes'],
    #     img_size=model_config['img_size'],
    #     patch_size=model_config['patch_size'],
    #     in_chans=model_config.get('in_channels', 3),
    #     embed_dim=model_config['embed_dim'],
    #     depth=model_config['depth'],
    #     num_heads=model_config['num_heads'],
    #     mlp_ratio=model_config.get('mlp_ratio', 4.0),
    #     drop_path_rate=model_config.get('drop_path_rate', 0.0),
    #     weight_init = 'jax_nlhb',
    #     # qk_norm = True,
    #     # init_values=model_config.get('layer_scale_init_value', 0.0),
    #     init_values=1e-6,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6)
    # )
    
        # 使用 timm.create_model 创建模型
    model = MAEVisionTransformer(
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
        # drop_path_rate=model_config.get('drop_path_rate', 0.0),
        drop_path_rate=0.1,
        weight_init = 'jax_nlhb',
        # qk_norm = True,
        # init_values=model_config.get('layer_scale_init_value', 0.0),
        init_values=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    
    return model

# def create_vit_te_model(config: Dict) -> VisionTransformerTE:
#     """
#     Factory function to create a VisionTransformerTE from a config dictionary.
#     """
#     model_config = config['model']
#     model = VisionTransformerTE(
#         img_size=model_config['img_size'],
#         patch_size=model_config['patch_size'],
#         in_channels=model_config.get('in_channels', 3),
#         embed_dim=model_config['embed_dim'],
#         depth=model_config['depth'],
#         num_heads=model_config['num_heads'],
#         mlp_ratio=model_config.get('mlp_ratio', 4.0),
#         num_classes=model_config['num_classes'],
#         dropout=model_config.get('dropout', 0.1)
#     )
#     return model

# # Example of how to instantiate and test the model, including backward pass
# # batch size, num_class all need to be 8/16 times.

# if __name__ == '__main__':
#     # Ensure CUDA is available
#     if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
#         device = torch.device("cuda")
#         batch_size = 16 # Example batch size
#         num_classes = 1000
        
#         # Define a model configuration
#         vit_b_16_config = {
#             'model': {
#                 'img_size': 224,
#                 'patch_size': 16,
#                 'embed_dim': 768,
#                 'depth': 12,
#                 'num_heads': 12,
#                 'num_classes': num_classes,
#                 'dropout': 0.1
#             }
#         }

#         # Instantiate the Transformer Engine ViT model using the factory
#         model = create_vit_te_model(vit_b_16_config).to(device)
#         # For a real training scenario, you would also wrap the model with DDP
#         # and torch.compile, e.g.:
#         # model = torch.nn.parallel.DistributedDataParallel(model)
#         # model = torch.compile(model)

#         print("Model Instantiated on CUDA with Transformer Engine layers via factory.")

#         # --- Verification Step ---
#         print("\n--- Running Verification: Forward and Backward Pass ---")
#         try:
#             # 1. Create dummy input and labels
#             dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
#             dummy_labels = torch.randint(0, num_classes, (batch_size,), device=device)
#             loss_fn = nn.CrossEntropyLoss()

#             # 2. Perform a forward pass
#             # The fp8_autocast is handled inside the model's forward method
#             output = model(dummy_input)
#             print(f"Forward pass successful!")
#             print(f"Output shape: {output.shape}") # Expected: [2, 1000]
#             print(f"Output dtype: {output.dtype}") # Expected: torch.float32

#             # 3. Calculate loss
#             # Note: The backward pass must be outside the fp8_autocast context
#             loss = loss_fn(output, dummy_labels)
#             print(f"Loss calculated: {loss.item():.4f}")

#             # 4. Perform a backward pass
#             loss.backward()
#             print("Backward pass successful!")

#             # 5. Verify gradients
#             # Check if the gradient of the final linear layer's weight exists
#             if model.head.weight.grad is not None:
#                 print("Gradient check PASSED. Gradients were computed for the head layer.")
#             else:
#                 print("Gradient check FAILED. No gradients found for the head layer.")
            
#             print("\nEnvironment and model code verified for a full training step. ✅")

#         except Exception as e:
#             print(f"\nAn error occurred during verification: {e}")
#             import traceback
#             traceback.print_exc()

#     else:
#         print("CUDA with Hopper architecture (FP8/BF16 support) not available. This model requires an H100 GPU.")
