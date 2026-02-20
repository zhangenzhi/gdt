import torch
import torch.nn as nn
from timm.layers import RotaryEmbedding, DropPath, Mlp
from timm.models.vision_transformer import VisionTransformer, Attention
from functools import partial
from typing import Dict


class SimpleRotary:
    """
    稳健的 RoPE 实现（不依赖 timm 的 RotaryEmbedding）。
    - 期望 dim (head_dim) 为偶数。
    - apply_rotary 接受 (BH, seq_len, dim) 的输入并返回相同形状输出。
    - 在内部用 float32 计算 sin/cos（更稳定），然后 cast 回输入 dtype。
    """
    def __init__(self, dim):
        assert dim % 2 == 0, "head_dim must be even for rotary"
        self.dim = dim
        half = dim // 2
        # inv_freq 存为 float32，延迟到 forward 时转 device/dtype
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
        self._inv_freq = inv_freq  # float32 tensor

    def _inv_freq_on(self, device, dtype):
        return self._inv_freq.to(device=device, dtype=torch.float32)  # keep float32 for stability

    def apply_rotary(self, x):
        """
        x: (BH, seq_len, dim)
        returns: same shape
        """
        assert x.ndim == 3, "apply_rotary expects (BH, seq_len, dim)"
        BH, seq_len, dim = x.shape
        assert dim == self.dim, f"expected last dim {self.dim}, got {dim}"
        half = dim // 2

        device = x.device
        in_dtype = x.dtype

        # compute pos frequencies in float32 (stable), then cast to in_dtype at use time
        inv_freq = self._inv_freq_on(device, in_dtype)  # (half,)
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)  # (seq_len,)
        freqs = torch.einsum("n,d->nd", pos, inv_freq)  # (seq_len, half)

        sin = freqs.sin().to(device=device, dtype=in_dtype)  # (seq_len, half)
        cos = freqs.cos().to(device=device, dtype=in_dtype)  # (seq_len, half)

        # split x into pairs: (BH, seq_len, half, 2)
        x_ = x.view(BH, seq_len, half, 2)
        x1 = x_[..., 0]  # (BH, seq_len, half)
        x2 = x_[..., 1]  # (BH, seq_len, half)

        # expand sin/cos for broadcast: (1, seq_len, half)
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)

        # rotation
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        out = torch.stack((rot_x1, rot_x2), dim=-1).view(BH, seq_len, dim)
        return out


class RopeAttention(Attention):
    """
    Attention with explicit, robust RoPE (SimpleRotary).
    使用方法：在构造时传入 dim 和 num_heads（和 timm Attention 接口一致），
    本类会自己创建 SimpleRotary(head_dim) 并在 forward 中以 (B*H, N, D) 形式调用。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, rope=None, **kwargs):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **kwargs)
        head_dim = dim // num_heads
        assert head_dim * num_heads == dim, "dim must be divisible by num_heads"
        # 如果外部传入 rope（可能是 timm 的 RotaryEmbedding），我们仍然用 SimpleRotary 替代以保证兼容性
        self.rope = SimpleRotary(head_dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, H, N, D)

        # 合并 batch 和 head -> (B*H, N, D)
        bh = B * self.num_heads
        # 使用 contiguous 保证内存布局安全
        q = q.reshape(bh, N, self.head_dim).contiguous()
        k = k.reshape(bh, N, self.head_dim).contiguous()
        v = v.reshape(bh, N, self.head_dim).contiguous()

        # 应用 RoPE
        q = self.rope.apply_rotary(q)
        k = self.rope.apply_rotary(k)

        # 恢复形状 -> (B, H, N, D)
        q = q.view(B, self.num_heads, N, self.head_dim)
        k = k.view(B, self.num_heads, N, self.head_dim)
        v = v.view(B, self.num_heads, N, self.head_dim)

        # attention 计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- 自定义 Block（替换 Attention）---
class RopeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RopeAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --- 主体 VisionTransformerWithRoPE ---
class VisionTransformerWithRoPE(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 **kwargs):
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
            **kwargs
        )

        # 禁用位置嵌入和 class token
        self.pos_embed = None
        self.cls_token = None
        self.global_pool = 'avg'

        # 实例化 RoPE
        head_dim = embed_dim // num_heads
        self.rope = RotaryEmbedding(dim=head_dim)

        # 重建 block，使用 RopeBlock
        self.blocks = nn.Sequential(*[
            RopeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=self.drop_path_rate * i / depth if hasattr(self, "drop_path_rate") else 0.,
                rope=self.rope
            )
            for i in range(depth)
        ])


# --- 工厂函数 ---
def create_rope_vit_model(config: Dict) -> VisionTransformerWithRoPE:
    model_config = config['model']
    model = VisionTransformerWithRoPE(
        img_size=model_config['img_size'],
        patch_size=model_config['patch_size'],
        in_chans=model_config.get('in_chans', 3),
        embed_dim=model_config['embed_dim'],
        depth=model_config['depth'],
        num_heads=model_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        num_classes=model_config['num_classes'],
        qkv_bias=model_config.get('qkv_bias', True),
    )
    return model