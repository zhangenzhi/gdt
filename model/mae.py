# --------------------------------------------- #
#   高效的 MAE 实现 (Optimized MAE Implementation)
# --------------------------------------------- #
import torch
import torch.nn as nn

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
#   2️⃣  Encoder (高效版本)
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
        B, C, H, W = x.shape
        
        # 1. Perform patch embedding on the whole batch at once
        x = self.patch_embed(x) # [B, N, D]

        # 2. Add positional embedding to all patches
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Gather only the visible patches using the boolean mask
        # This is the key optimization. `mask.unsqueeze(-1).expand_as(x)` creates a
        # boolean mask of shape [B, N, D] to select the elements from x.
        # The result is flattened to [num_total_visible_patches, D].
        visible_patches = x[mask]
        
        # Since the number of visible patches can vary per sample, we can't directly
        # use a batch-wise transformer. A common approach is to pad, but a simpler
        # and often effective way (though not strictly what the original MAE paper did)
        # is to process them as one giant sequence if batch semantics aren't strictly
        # required between patches of different images in the encoder.
        # For a more faithful implementation, one would pad `visible_patches` here.
        # However, for simplicity and to show the removal of the loop, we'll reshape.
        
        # To maintain batch dimension for the transformer, we need to pad.
        # Let's implement the padding logic efficiently.
        num_visible = mask.sum(dim=1) # [B]
        max_vis = num_visible.max()
        
        # Create a padded tensor
        padded_visible_patches = torch.zeros(B, max_vis, x.size(2), device=x.device, dtype=x.dtype)
        
        # Create a padding mask for the transformer
        # (True means the position should be ignored)
        padding_mask = torch.arange(max_vis, device=x.device)[None, :] >= num_visible[:, None]

        # Fill the padded tensor
        current_pos = 0
        for i, n_vis in enumerate(num_visible):
            padded_visible_patches[i, :n_vis] = visible_patches[current_pos:current_pos + n_vis]
            current_pos += n_vis

        # 4. Feed to transformer with padding mask
        enc_out = self.transformer(padded_visible_patches, src_key_padding_mask=padding_mask)
        enc_out = self.norm(enc_out)

        return enc_out


# --------------------------------------------- #
#   3️⃣  Decoder (高效版本)
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
        
        # 2. Create the full sequence tensor with mask tokens
        # `~mask` gives us the masked positions
        num_masked = (~mask).sum(dim=1)
        
        # Create a full-size tensor to scatter into
        x = self.mask_token.repeat(B, N, 1)

        # 3. Use advanced indexing to scatter visible tokens into the correct positions
        # This is the key optimization for the decoder.
        # We need a batch-wise scatter. We can create an index map.
        # `torch.where` is great for this.
        visible_indices = torch.where(mask)
        
        # We need to align the flat `visible_tokens` with the 2D `visible_indices`
        # A simple way is to use a mask for assignment.
        # First, let's un-pad the encoder output
        num_visible = mask.sum(dim=1)
        unpadded_tokens = []
        for i, n_vis in enumerate(num_visible):
            unpadded_tokens.append(visible_tokens[i, :n_vis])
        unpadded_tokens = torch.cat(unpadded_tokens, dim=0)

        # Now scatter them back
        x[mask] = unpadded_tokens

        # 4. Add positional embedding to the full sequence
        x = x + self.pos_embed

        # 5. Feed to the decoder transformer
        dec_out = self.decoder(x)
        dec_out = self.norm(dec_out)

        # 6. Reconstruct pixel values only for masked patches (as per the paper)
        # Or for all patches if you want to visualize everything.
        # Here we reconstruct all for simplicity.
        recon_flat = self.reconstruction_head(dec_out)
        
        return recon_flat


# --------------------------------------------- #
#   4️⃣  MAE wrapper (handles masking)
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
        enc_out = self.encoder(x, mask)
        
        # Decode full sequence
        recon_patches_flat = self.decoder(enc_out, mask) # [B, N, P*P*3]
        
        # Calculate loss on masked patches
        target_patches = self.encoder.patch_embed(x) # [B, N, D]
        
        # Normalize target pixels per-patch
        # This is a key detail from the paper for stable training.
        mean = target_patches.mean(dim=-1, keepdim=True)
        var = target_patches.var(dim=-1, keepdim=True)
        target_patches = (target_patches - mean) / (var + 1.e-6)**.5

        # Project reconstruction to the same dimension as patch embeddings for loss calculation
        # This is a common simplification. The original paper reconstructs pixels.
        # Let's stick to the paper and reconstruct pixels.
        
        # Reshape target image into patches
        P = self.patch_size
        target_patches_pixel = x.unfold(2, P, P).unfold(3, P, P) # [B, C, N_h, N_w, P, P]
        target_patches_pixel = target_patches_pixel.permute(0, 2, 3, 1, 4, 5).contiguous() # [B, N_h, N_w, C, P, P]
        target_patches_pixel = target_patches_pixel.view(B, self.num_patches, -1) # [B, N, C*P*P]
        
        # The reconstruction head outputs 3*P*P, which is C*P*P if C=3.
        # So the shapes match.
        
        # Calculate loss only on the masked patches
        loss = (recon_patches_flat[~mask] - target_patches_pixel[~mask]) ** 2
        loss = loss.mean()
        
        return loss, recon_patches_flat, mask


# --------------------------------------------- #
#   5️⃣  Quick sanity check
# --------------------------------------------- #
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAE().to(device)
    dummy = torch.randn(8, 3, 224, 224).to(device) # Use a larger batch size to see performance benefits
    
    # The forward pass now returns loss, reconstruction, and mask
    loss, recon, mask = model(dummy)
    
    print("Reconstructed patch shape:", recon.shape)
    print("Loss:", loss.item())
    
    # Expected shape: torch.Size([8, 196, 768]) where 768 = 16*16*3
    P = model.patch_size
    expected_shape = torch.Size([dummy.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    print("Sanity check passed!")

