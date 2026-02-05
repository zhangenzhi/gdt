import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

# --- 1. Spatio-Structural Positional Embedding ---
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

# --- 2. HMAE Encoder (Processes only visible patches) ---
class HMAEEncoder(nn.Module):
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SpatioStructuralPosEmbed(embed_dim, img_size)
        
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, coords, depths, ids_keep):
        # x: [B, N, patch_dim]
        B = x.shape[0]
        
        # 1. Patch embedding
        x = self.patch_embed(x)
        
        # 2. Add structural pos embed (to all patches first)
        x = x + self.pos_embed(coords, depths)
        
        # 3. Gather visible patches
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        # 4. Prepend cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        # Note: CLS token doesn't have a specific spatial coordinate, we add it without pos_embed or with a dummy
        x = torch.cat((cls_token, x), dim=1)
        
        # 5. Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x

# --- 3. HMAE Decoder (Reconstructs full sequence) ---
class HMAEDecoder(nn.Module):
    def __init__(self, img_size=1024, patch_size=32, in_channels=1, encoder_dim=768, decoder_dim=512, depth=8, num_heads=16):
        super().__init__()
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (img_size // patch_size) ** 2 # Placeholder, actual used is from processor
        
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed = SpatioStructuralPosEmbed(decoder_dim, img_size)
        
        self.blocks = nn.ModuleList([
            Block(dim=decoder_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, self.patch_dim, bias=True)

    def forward(self, x, coords, depths, ids_restore):
        # 1. Embed to decoder dimension
        x = self.decoder_embed(x)

        # 2. Prepare full sequence tokens
        # x: [B, n_visible + 1, D] (the +1 is CLS)
        B, N_full = ids_restore.shape[0], ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(B, N_full - (x.shape[1] - 1), 1)
        
        # Concatenate visible tokens (minus CLS) with mask tokens
        x_shuffled = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        # Un-shuffle using ids_restore
        x_full = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        # 3. Add decoder structural pos embed
        x_full = x_full + self.pos_embed(coords, depths)
        
        # 4. Decoder blocks
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)
        
        # 5. Predict pixels
        pred = self.head(x_full)
        return pred

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
        # targets: [B, N, C, P, P]
        target = targets.flatten(2)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var.sqrt() + 1e-6)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def forward(self, patches, coords, depths):
        # patches: [B, N, C, P, P]
        B, N, C, PH, PW = patches.shape
        x = patches.flatten(2)
        
        # 1. Masking
        ids_keep, mask, ids_restore = self.random_masking(B, N, x.device)
        
        # 2. Encoding (Visible only)
        enc_out = self.encoder(x, coords, depths, ids_keep)
        
        # 3. Decoding (Full sequence)
        pred = self.decoder(enc_out, coords, depths, ids_restore)
        
        # 4. Loss
        loss = self.forward_loss(patches, pred, mask)
        
        return loss, pred, mask