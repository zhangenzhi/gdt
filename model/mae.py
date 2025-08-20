# --------------------------------------------- #
#   高效的 MAE 实现 (Optimized MAE Implementation)
# --------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# A utility for padding variable-length sequences
from torch.nn.utils.rnn import pad_sequence

# --------------------------------------------- #
#   MAE 模型定义 (from previous Canvas)
# --------------------------------------------- #
class PatchEmbedding(nn.Module):
    """Image → Patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)

class MAEEncoder(nn.Module):
    """Encoder that processes only visible patches. (Efficient Version)"""
    def __init__(self, *, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        visible_patches_list = [x[i][mask[i]] for i in range(B)]
        padded_visible_patches = pad_sequence(visible_patches_list, batch_first=True)
        num_visible = mask.sum(dim=1)
        max_vis = num_visible.max()
        padding_mask = torch.arange(max_vis, device=x.device)[None, :] >= num_visible[:, None]
        enc_out = self.transformer(padded_visible_patches, src_key_padding_mask=padding_mask)
        enc_out = self.norm(enc_out)
        return enc_out, visible_patches_list

class MAEDecoder(nn.Module):
    """Decoder that reconstructs all patches. (Efficient Version)"""
    def __init__(self, *, embed_dim=768, decoder_depth=4,
                 decoder_heads=16, decoder_dim=512, patch_size=16,
                 img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_dim = decoder_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer,
                                             num_layers=decoder_depth)
        self.reconstruction_head = nn.Linear(decoder_dim,
                                             patch_size * patch_size * 3, bias=True)
        self.norm = nn.LayerNorm(decoder_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, enc_out: torch.Tensor, mask: torch.BoolTensor):
        B, N = mask.shape
        visible_tokens = self.decoder_embed(enc_out)
        unpadded_tokens_list = [visible_tokens[i, :mask.sum(dim=1)[i]] for i in range(B)]
        flat_unpadded_tokens = torch.cat(unpadded_tokens_list, dim=0)
        
        # FIX: Cast the mask_token to the same dtype as the encoder output
        full_sequence = self.mask_token.to(enc_out.dtype).repeat(B, N, 1)
        
        full_sequence[mask] = flat_unpadded_tokens
        
        # FIX: Also cast the positional embedding to the same dtype
        full_sequence = full_sequence + self.pos_embed.to(enc_out.dtype)
        
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat

class MAE(nn.Module):
    """End-to-end Masked Auto-Encoder."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_embed_dim=512,
                 decoder_depth=4,
                 decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        self.decoder = MAEDecoder(
            embed_dim=encoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            decoder_dim=decoder_embed_dim,
            patch_size=patch_size,
            img_size=img_size,
        )

    def random_masking(self, B: int, device: torch.device) -> torch.BoolTensor:
        """Random binary mask – True = visible patch"""
        return torch.rand(B, self.num_patches, device=device) < (1.0 - self.mask_ratio)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        mask = self.random_masking(B, x.device)
        enc_out, _ = self.encoder(x, mask)
        recon_patches_flat = self.decoder(enc_out, mask)
        P = self.patch_size
        target_patches_pixel = F.unfold(x, kernel_size=P, stride=P).transpose(1, 2)
        loss = (recon_patches_flat[~mask] - target_patches_pixel[~mask]) ** 2
        loss = loss.mean()
        # Return loss and all components needed for visualization
        return loss, recon_patches_flat, mask, target_patches_pixel


# --------------------------------------------- #
#   5️⃣  Quick sanity check
# --------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE().to(device)
    dummy = torch.randn(512, 3, 224, 224).to(device)
    
    # The forward pass now returns loss, reconstruction, and mask
    loss, recon, mask = model(dummy)
    
    print("Reconstructed patch shape:", recon.shape)
    print("Loss:", loss.item())
    
    P = model.patch_size
    expected_shape = torch.Size([dummy.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    print("Sanity check passed!")

