import torch
from torch import nn
from typing import List, Dict

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

class VisionTransformer(nn.Module):
    """Standard Vision Transformer with a Transformer Encoder."""
    def __init__(self, *, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, num_classes=1000, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)
        
        # Get the CLS token for classification
        cls_output = self.norm(x[:, 0])
        logits = self.head(cls_output)
        
        # Return only logits to match standard classifier output
        return logits

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
        num_classes=model_config['num_classes']
    )
    return model