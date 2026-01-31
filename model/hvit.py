import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, VisionTransformer

class SpatioStructuralPosEmbed(nn.Module):
    def __init__(self, embed_dim, img_size=1024, max_depth=8):
        super().__init__()
        self.img_size = float(img_size)
        self.max_depth = float(max_depth)
        self.proj = nn.Sequential(
            nn.Linear(5, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, coords, depths):
        x1, x2, y1, y2 = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        features = torch.stack([
            cx / self.img_size, cy / self.img_size, 
            w / self.img_size, h / self.img_size, 
            depths / self.max_depth
        ], dim=-1)
        return self.proj(features)

class HMAEVIT(nn.Module):
    """
    接收预处理后的 Patch 张量进行训练。不再在内部处理 Quadtree。
    """
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, encoder_dim=768, 
                 encoder_depth=12, encoder_heads=12, decoder_dim=512, decoder_depth=8, 
                 decoder_heads=16):
        super().__init__()
        self.patch_size = patch_size
        
        # Encoder
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)
        self.pos_embed = SpatioStructuralPosEmbed(encoder_dim, img_size)
        
        timm_v = VisionTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_channels,
                                  num_classes=0, global_pool='', embed_dim=encoder_dim,
                                  depth=encoder_depth, num_heads=encoder_heads)
        self.encoder_blocks = timm_v.blocks
        self.encoder_norm = timm_v.norm

        # Decoder
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = SpatioStructuralPosEmbed(decoder_dim, img_size)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=decoder_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        self.head_image = nn.Linear(decoder_dim, patch_dim)
        self.head_noise = nn.Linear(decoder_dim, patch_dim)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, patches, coords, depths, mask):
        # 1. Encoding
        x = patches.flatten(2)
        x = self.patch_embed(x) + self.pos_embed(coords, depths)
        
        B, L, D = x.shape
        visible_mask = (mask == 0)
        encoded_full = torch.zeros(B, L, D, device=x.device)
        
        # 仅编码可见 Patch
        for b in range(B):
            v_idx = visible_mask[b]
            x_vis = x[b:b+1, v_idx, :]
            if x_vis.size(1) > 0:
                x_vis = self.encoder_blocks(x_vis)
                encoded_full[b, v_idx, :] = self.encoder_norm(x_vis)[0]

        # 2. Decoding
        x_dec = self.decoder_embed(encoded_full)
        mask_tokens = self.mask_token.repeat(B, L, 1)
        x_dec = torch.where((mask != 0).unsqueeze(-1), mask_tokens, x_dec)
        x_dec = x_dec + self.decoder_pos_embed(coords, depths)
        
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec)
        x_dec = self.decoder_norm(x_dec)

        # 3. Prediction
        pred_img = self.head_image(x_dec)
        pred_noise = self.head_noise(x_dec)
        
        return pred_img, pred_noise