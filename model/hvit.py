import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from timm.models.vision_transformer import Block

# # --- 1. Spatio-Structural Positional Embedding ---
# class SpatioStructuralPosEmbed(nn.Module):
#     def __init__(self, embed_dim, img_size=1024, max_depth=8):
#         super().__init__()
#         self.img_size = float(img_size)
#         self.max_depth = float(max_depth)
#         self.proj = nn.Sequential(
#             nn.Linear(5, embed_dim // 2),
#             nn.GELU(),
#             nn.Linear(embed_dim // 2, embed_dim)
#         )

#     def forward(self, coords, depths):
#         x1, x2, y1, y2 = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
#         cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#         w, h = x2 - x1, y2 - y1
#         features = torch.stack([
#             cx / self.img_size, cy / self.img_size, 
#             w / self.img_size, h / self.img_size, 
#             depths / self.max_depth
#         ], dim=-1)
#         return self.proj(features)


# --- 1. RoPE 核心逻辑 ---

def rotate_half(x):
    """将张量最后维度对半分并交换，用于旋转计算。"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """将旋转位置编码应用于 Query 和 Key。"""
    # q, k: [B, num_heads, L, head_dim]
    # cos, sin: [B, 1, L, head_dim]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class HMAERotaryEmbedding(nn.Module):
    """
    针对 2D 连续坐标的旋转位置编码。
    将 head_dim 分成两部分，分别对应 x 和 y 的旋转。
    """
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        # 预计算频率标量
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords):
        """
        coords: [B, N, 2] (归一化后的中心点坐标 cx, cy)
        返回: cos, sin 形状为 [B, 1, N, head_dim]
        """
        # 分离 x 和 y 坐标
        cx = coords[..., 0:1] # [B, N, 1]
        cy = coords[..., 1:2] # [B, N, 1]

        # 计算旋转弧度
        # x_freq, y_freq: [B, N, head_dim // 4]
        x_freq = cx * self.inv_freq
        y_freq = cy * self.inv_freq

        # 构造完整的 head 频率序列
        # 每个坐标占据 head_dim 的一半
        # 最终拼接成 [B, N, head_dim // 2]
        all_freqs = torch.cat([x_freq, y_freq], dim=-1)
        
        # 复制一份以匹配 rotate_half 的逻辑: [sin, sin] 和 [cos, cos]
        emb = torch.cat((all_freqs, all_freqs), dim=-1) # [B, N, head_dim]
        
        cos = emb.cos().unsqueeze(1) # [B, 1, N, head_dim]
        sin = emb.sin().unsqueeze(1) # [B, 1, N, head_dim]
        return cos, sin

# --- 2. 适配 PyTorch 原生 SDPA (FlashAttention) ---

class HMAEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_prob = attn_drop # 供 SDPA 使用
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, N, C = x.shape
        
        # 1. 生成 QKV
        # 变换为 PyTorch 标准形状: [B, H, N, D]
        # reshape: [B, N, 3, H, D] -> permute: [3, B, H, N, D]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, H, N, D]

        # 2. 应用 RoPE
        # rope_cos/sin 形状为 [B, 1, N, D]，自动广播到 H 维度
        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # 3. 使用 PyTorch 原生 FlashAttention (SDPA)
        # 自动选择 FlashAttention V2 或 Memory Efficient Attention
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            scale=self.scale
        )
        
        # 4. 重塑回 [B, N, C]
        # [B, H, N, D] -> [B, N, H, D] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HMAEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = HMAEAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x, rope_cos=None, rope_sin=None):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.norm2(x))
        return x
    
# --- 3. HMAE Encoder & Decoder (集成 RoPE) ---

class HMAEEncoder(nn.Module):
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # RoPE 发生器
        self.rope = HMAERotaryEmbedding(head_dim=embed_dim // num_heads)
        
        # 结构信息补偿：由于 RoPE 擅长位置，但无法直接表达 depth 和 w/h，
        # 我们仍保留一个轻量的 MLP 来嵌入这些“非位置”的结构属性。
        self.struct_embed = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

        self.blocks = nn.ModuleList([
            HMAEBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, coords, depths, ids_keep):
        B, N, _ = x.shape
        
        # 1. Patch 内容嵌入
        x = self.patch_embed(x)
        
        # # 2. 注入非位置结构信息 (w, h, depth)
        # w = (coords[..., 1] - coords[..., 0]) / self.img_size
        # h = (coords[..., 3] - coords[..., 2]) / self.img_size
        # s_feat = torch.stack([w, h, depths / 8.0], dim=-1)
        # x = x + self.struct_embed(s_feat)

        # # 3. 准备 RoPE
        # cx = (coords[..., 0] + coords[..., 1]) / (2.0 * self.img_size)
        # cy = (coords[..., 2] + coords[..., 3]) / (2.0 * self.img_size)
        # c_feat = torch.stack([cx, cy], dim=-1)
        # cos, sin = self.rope(c_feat)

        # --- 1. 结构嵌入 (struct_embed) ---
        # 这里的 w, h, depth 保持归一化是没问题的，因为它们进的是 MLP (Linear)
        w = (coords[..., 1] - coords[..., 0]) / self.img_size
        h = (coords[..., 3] - coords[..., 2]) / self.img_size
        s_feat = torch.stack([w, h, depths / 8.0], dim=-1)
        x = x + self.struct_embed(s_feat)

        # --- 2. RoPE 准备 (关键修改！！！) ---
        # 错误写法 (导致收敛慢): 
        # cx = (coords[..., 0] + coords[..., 1]) / (2.0 * self.img_size)
        
        # 正确写法: 使用绝对像素坐标，或者是相对 Patch Size 的坐标
        # 例如: 如果 img_size=1024，cx 范围就是 0~1024
        # 这样配合 theta=10000，才能产生足够显著的旋转差异
        cx = (coords[..., 0] + coords[..., 1]) / 2.0 
        cy = (coords[..., 2] + coords[..., 3]) / 2.0
        
        # 此时 cx, cy 是 float32，范围 [0, 1024]
        c_feat = torch.stack([cx, cy], dim=-1)
        
        # 生成 RoPE
        cos, sin = self.rope(c_feat)
        
        # 4. Gather 选中的补丁及其对应的 RoPE
        # x: [B, len_keep, D]
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        # cos, sin: [B, 1, len_keep, head_dim]
        cos = torch.gather(cos, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, cos.shape[-1]))
        sin = torch.gather(sin, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, sin.shape[-1]))

        # 5. 添加 CLS Token (不使用 RoPE)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # CLS 对应的 cos/sin 设为 1.0/0.0 (不旋转)
        cls_cos = torch.ones(B, 1, 1, cos.shape[-1], device=x.device)
        cls_sin = torch.zeros(B, 1, 1, sin.shape[-1], device=x.device)
        cos = torch.cat([cls_cos, cos], dim=2)
        sin = torch.cat([cls_sin, sin], dim=2)

        for blk in self.blocks:
            x = blk(x, cos, sin)
        
        return self.norm(x)

class HMAEDecoder(nn.Module):
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, encoder_dim=768, decoder_dim=512, depth=8, num_heads=16):
        super().__init__()
        self.img_size = img_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.rope = HMAERotaryEmbedding(head_dim=decoder_dim // num_heads)
        self.struct_embed = nn.Sequential(
            nn.Linear(3, decoder_dim // 4),
            nn.GELU(),
            nn.Linear(decoder_dim // 4, decoder_dim)
        )

        self.blocks = nn.ModuleList([
            HMAEBlock(dim=decoder_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, self.patch_dim, bias=True)

    def forward(self, x, coords, depths, ids_restore):
        x = self.decoder_embed(x)
        B, N_full = ids_restore.shape[0], ids_restore.shape[1]
        
        # 填充 Mask Tokens
        mask_tokens = self.mask_token.repeat(B, N_full - (x.shape[1] - 1), 1)
        x_shuffled = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_full = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        # 结构信息注入
        w = (coords[..., 1] - coords[..., 0]) / self.img_size
        h = (coords[..., 3] - coords[..., 2]) / self.img_size
        s_feat = torch.stack([w, h, depths / 8.0], dim=-1)
        x_full = x_full + self.struct_embed(s_feat)

        # 准备全序列 RoPE
        cx = (coords[..., 0] + coords[..., 1]) / (2.0 * self.img_size)
        cy = (coords[..., 2] + coords[..., 3]) / (2.0 * self.img_size)
        c_feat = torch.stack([cx, cy], dim=-1)
        cos, sin = self.rope(c_feat)

        for blk in self.blocks:
            x_full = blk(x_full, cos, sin)
        
        x_full = self.norm(x_full)
        return self.head(x_full)

# --- 4. HMAEVIT Wrapper ---
class HMAEVIT(nn.Module):
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, 
                 encoder_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_dim=512, decoder_depth=8, decoder_heads=16,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        self.encoder = HMAEEncoder(img_size, patch_size, in_channels, encoder_dim, encoder_depth, encoder_heads)
        self.decoder = HMAEDecoder(img_size, patch_size, in_channels, encoder_dim, decoder_dim, decoder_depth, decoder_heads)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, B, N, device):
        """MAE-style random masking logic."""
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        mask = torch.ones([B, N], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, mask, ids_restore

    def forward_loss(self, targets, pred, mask):
        target = targets.flatten(2)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var.sqrt() + 1e-6)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, patches, coords, depths):
        B, N, C, PH, PW = patches.shape
        x = patches.flatten(2)
        
        ids_keep, mask, ids_restore = self.random_masking(B, N, x.device)
        enc_out = self.encoder(x, coords, depths, ids_keep)
        pred = self.decoder(enc_out, coords, depths, ids_restore)
        loss = self.forward_loss(patches, pred, mask)
        
        return loss, pred, mask

# --- 5. 模型定义 (B, L, XL) ---

def hvit_b(**kwargs):
    return HMAEVIT(
        encoder_dim=768, encoder_depth=12, encoder_heads=12,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        **kwargs)

def hvit_l(**kwargs):
    return HMAEVIT(
        encoder_dim=1024, encoder_depth=24, encoder_heads=16,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        **kwargs)

def hvit_xl(**kwargs):
    return HMAEVIT(
        encoder_dim=1280, encoder_depth=32, encoder_heads=16,
        decoder_dim=512, decoder_depth=8, decoder_heads=16,
        **kwargs)