import torch
import torch.nn as nn
import torch.nn.functional as F

# A utility for padding variable-length sequences
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast # Import autocast for bfloat16 testing

# --------------------------------------------- #
#   1️⃣  Patch embedding (unchanged)
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
        # [B, C, H, W] → [B, D, H/P, W/P] → [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


# --------------------------------------------- #
#   2️⃣  Encoder (高效版本 - 已修复)
# --------------------------------------------- #
class MAEEncoder(nn.Module):
    """Encoder that processes only visible patches. (Efficient Version)"""
    def __init__(self, *, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding for ALL patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True # Common practice for stability
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
        """
        Efficiently processes only the visible patches without loops.
        
        Parameters
        ----------
        x    : [B, C, H, W]
        mask : [B, N]  (True = visible patch)

        Returns
        -------
        enc_out   : [B, num_visible, D]
        """
        B, _, _, _ = x.shape
        
        # 1. Perform patch embedding on the whole batch at once
        x = self.patch_embed(x) # [B, N, D]

        # 2. Add positional embedding to all patches
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Gather only the visible patches using the boolean mask, creating a list of tensors.
        # This is more robust than a flattened tensor.
        visible_patches_list = [x[i][mask[i]] for i in range(B)]
        
        # 4. Use pad_sequence to pad the list of tensors to the same size
        # This is a highly efficient, vectorized operation.
        padded_visible_patches = pad_sequence(visible_patches_list, batch_first=True)
        
        # 5. Create a padding mask for the transformer
        num_visible = mask.sum(dim=1)
        max_vis = num_visible.max()
        padding_mask = torch.arange(max_vis, device=x.device)[None, :] >= num_visible[:, None]

        # 6. Feed to transformer with padding mask
        enc_out = self.transformer(padded_visible_patches, src_key_padding_mask=padding_mask)
        enc_out = self.norm(enc_out)

        # Return unpadded encoder output for the decoder
        # The decoder will handle the padding internally
        return enc_out, visible_patches_list


# --------------------------------------------- #
#   3️⃣  Decoder (高效版本 - 已修复)
# --------------------------------------------- #
class MAEDecoder(nn.Module):
    """Decoder that reconstructs all patches. (Efficient Version)"""
    def __init__(self, *, embed_dim=768, decoder_depth=4,
                 decoder_heads=16, decoder_dim=512, patch_size=16,
                 img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_dim = decoder_dim

        # Learned token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Project encoder outputs (D_enc → D_dec)
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)

        # Positional embedding for the full sequence
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

        # Final projection to pixel values (3 × P × P)
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
        """
        Efficiently reconstructs the full image without loops.

        Parameters
        ----------
        enc_out : [B, num_visible, D_enc] (padded)
        mask    : [B, N]                  (True = visible)

        Returns
        -------
        recon   : [B, N, 3*P*P]
        """
        B, N = mask.shape
        
        # 1. Project encoder outputs to decoder dimension
        visible_tokens = self.decoder_embed(enc_out) # [B, num_visible, D_dec]
        
        # 2. Flatten the visible tokens to prepare for scatter
        unpadded_tokens_list = [visible_tokens[i, :mask.sum(dim=1)[i]] for i in range(B)]
        flat_unpadded_tokens = torch.cat(unpadded_tokens_list, dim=0)
        
        # 3. Create the full sequence tensor with mask tokens
        full_sequence = self.mask_token.to(visible_tokens.dtype).repeat(B, N, 1)

        # 4. Scatter the unpadded tokens (保证 dtype 一致)
        full_sequence[mask] = flat_unpadded_tokens.to(visible_tokens.dtype)

        # 5. Add positional embedding
        full_sequence = full_sequence + self.pos_embed.to(visible_tokens.dtype)

        # 6. Feed to the decoder transformer
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)

        # 7. Reconstruct pixel values
        recon_flat = self.reconstruction_head(dec_out)
        
        return recon_flat


# --------------------------------------------- #
#   4️⃣  MAE wrapper (handles masking - 已修复)
# --------------------------------------------- #
class MAE(nn.Module):
    """
    End-to-end Masked Auto-Encoder.
    """
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
        """
        Random binary mask – True = visible patch
        """
        return torch.rand(B, self.num_patches, device=device) < (1.0 - self.mask_ratio)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : [B, C, H, W]

        Returns
        -------
        loss    : The reconstruction loss (scalar).
        recon   : [B, N, P*P*3], the reconstructed patches.
        mask    : [B, N], the mask used.
        """
        B = x.shape[0]
        mask = self.random_masking(B, x.device) # [B, N]  True = visible
        
        # Encode only visible patches
        # NOTE: The encoder now returns padded output AND a list of unpadded tokens
        enc_out, _ = self.encoder(x, mask)
        
        # Decode full sequence
        recon_patches_flat = self.decoder(enc_out, mask) # [B, N, P*P*3]
        
        # Calculate loss on masked patches
        # Using F.unfold is a cleaner way to extract patches
        P = self.patch_size
        target_patches_pixel = F.unfold(x, kernel_size=P, stride=P).transpose(1, 2) # [B, N, C*P*P]
        
        # Calculate loss only on the masked patches
        # The reconstruction head outputs 3*P*P, which matches C*P*P if C=3.
        loss = (recon_patches_flat[~mask] - target_patches_pixel[~mask]) ** 2
        loss = loss.mean()
        
        return loss, recon_patches_flat, mask


# --------------------------------------------- #
#   5️⃣  Quick sanity check
# --------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE().to(device)
    
    # Use bfloat16 for the dummy tensor
    # NOTE: The model and input must be on a CUDA device to use bfloat16
    if device.type == "cuda":
        dummy = torch.randn(8, 3, 224, 224, device=device).to(torch.bfloat16)
    else:
        # Fallback to float32 if not on CUDA, as bfloat16 is a GPU-specific feature
        dummy = torch.randn(8, 3, 224, 224, device=device)

    # Use autocast to run the forward pass in bfloat16
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        # The forward pass now returns loss, reconstruction, and mask
        loss, recon, mask = model(dummy)
    
    print("Reconstructed patch shape:", recon.shape)
    print("Reconstructed patch dtype:", recon.dtype)
    print("Loss:", loss.item())
    
    P = model.patch_size
    expected_shape = torch.Size([dummy.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    assert recon.dtype == torch.bfloat16, f"Dtype mismatch! Got {recon.dtype}, expected {torch.bfloat16}"
    print("Sanity check passed!")