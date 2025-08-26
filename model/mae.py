# --------------------------------------------- #
#   File: mae.py
#   MAE Model Definition (Revised and Annotated)
# --------------------------------------------- #
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast # REVISED: Updated import for modern API

# --------------------------------------------- #
#   1️⃣  Patch embedding (No changes)
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
#   2️⃣  Encoder (No changes to logic)
# --------------------------------------------- #
class MAEEncoder(nn.Module):
    """Encoder that processes only visible patches."""
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

        # NOTE: The list comprehension and pad_sequence are a standard way to
        # handle variable-length sequences for the transformer, but can be a
        # bottleneck for very large batches. For most cases, it's clear and correct.
        visible_patches_list = [x[i][~mask[i]] for i in range(B)]
        padded_visible_patches = pad_sequence(visible_patches_list, batch_first=True)

        # Create the padding mask for the transformer's attention
        num_visible = (~mask).sum(dim=1)
        max_vis = num_visible.max()
        # True indicates a position that should be ignored by attention.
        padding_mask = torch.arange(max_vis, device=x.device)[None, :] >= num_visible[:, None]

        enc_out = self.transformer(padded_visible_patches, src_key_padding_mask=padding_mask)
        enc_out = self.norm(enc_out)
        return enc_out

# --------------------------------------------- #
#   3️⃣  Decoder (Logic Refactored for Clarity)
# --------------------------------------------- #
class MAEDecoder(nn.Module):
    """Decoder that reconstructs all patches."""
    def __init__(self, *, encoder_dim=768, decoder_dim=512, decoder_depth=4,
                 decoder_heads=16, patch_size=16, img_size=224):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.reconstruction_head = nn.Linear(decoder_dim, patch_size**2 * 3, bias=True)
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

    # --- REVISED FORWARD PASS ---
    def forward(self, enc_out: torch.Tensor, mask: torch.BoolTensor):
        """
        Reconstructs the full sequence of patches from the encoded visible patches.

        Args:
            enc_out (torch.Tensor): Output from the encoder [B, L_vis_max, D_enc].
                                    It's padded to the max number of visible tokens.
            mask (torch.BoolTensor): The mask used in the encoder [B, N].
                                     True = masked, False = visible.
        """
        # 1. Embed encoder tokens to the decoder's dimension
        # [B, L_vis_max, D_enc] -> [B, L_vis_max, D_dec]
        visible_tokens = self.decoder_embed(enc_out)
        B, N = mask.shape
        D_dec = visible_tokens.size(-1)

        # 2. Create a placeholder for the full sequence, filled with mask tokens
        # [B, N, D_dec]
        full_sequence = self.mask_token.expand(B, N, D_dec).clone()

        # 3. Unpad the encoder output to get a flat list of valid visible tokens
        num_visible = (~mask).sum(dim=1)  # [B]
        max_vis = visible_tokens.size(1)  # L_vis_max
        # Create a boolean mask to identify valid (non-padding) tokens
        keep_mask = torch.arange(max_vis, device=enc_out.device)[None, :] < num_visible[:, None] # [B, L_vis_max]
        # Flatten and select only the valid tokens
        valid_visible_tokens = visible_tokens[keep_mask] # [total_visible_patches, D_dec]

        # 4. Get the original positions of the visible patches
        # `torch.where` gives us the coordinates of all `False` (visible) entries in the mask.
        batch_indices, patch_indices = torch.where(~mask)

        # 5. Scatter the visible tokens into their original positions
        # This is the core of the reconstruction. We place the `valid_visible_tokens`
        # into the `full_sequence` placeholder at their original locations.
        full_sequence[batch_indices, patch_indices] = valid_visible_tokens

        # 6. Add positional embeddings to the complete sequence
        full_sequence = full_sequence + self.pos_embed

        # 7. Pass the full sequence through the decoder
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)

        # 8. Project to pixel space for reconstruction
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat


# --------------------------------------------- #
#   4️⃣  MAE wrapper (Parameter names clarified)
# --------------------------------------------- #
class MAE(nn.Module):
    """End-to-end Masked Auto-Encoder."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512, # REVISED: Clearer parameter name
                 decoder_depth=4,
                 decoder_heads=16,
                 mask_ratio=0.75,
                 norm_pix_loss=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim, # REVISED: Pass correct dimensions
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_size=patch_size,
            img_size=img_size,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Converts an image into a sequence of flattened patches.
        imgs: (B, C, H, W) -> x: (B, N, P*P*C)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0, "Image dimensions must be divisible by patch size."
        num_patches_h = H // p
        # (B, C, H, W) -> (B, C, h, p, w, p) -> (B, h, w, C, p, p) -> (B, h*w, C*p*p)
        x = imgs.reshape(B, C, num_patches_h, p, num_patches_h, p)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B, self.num_patches, C * p * p)
        return x

    def random_masking(self, B: int, device: torch.device) -> torch.BoolTensor:
        """
        Generates a random binary mask. True = masked, False = keep.
        """
        return torch.rand(B, self.num_patches, device=device) < self.mask_ratio

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Calculates the reconstruction loss on the masked patches.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], loss per patch

        # Only consider loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x: torch.Tensor):
        """
        Full forward pass of the MAE model.
        """
        B = x.shape[0]
        # True = masked, False = visible
        mask = self.random_masking(B, x.device)

        # Encode only the visible patches
        enc_out = self.encoder(x, mask)

        # Decode the full sequence of patches
        recon_patches_flat = self.decoder(enc_out, mask)

        # Calculate the reconstruction loss
        loss = self.forward_loss(x, recon_patches_flat, mask)

        return loss, recon_patches_flat, mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use bfloat16 on CUDA if available, otherwise float32
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")

    model = MAE().to(device)
    dummy_input = torch.randn(8, 3, 224, 224, device=device)

    # Use autocast for mixed-precision training
    # REVISED: Corrected the autocast call to match the modern PyTorch API.
    # The device type is now a positional argument, not a keyword argument.
    with autocast(device.type, dtype=dtype):
        loss, recon, mask = model(dummy_input)

    # Simple backward pass to check for gradient flow
    loss.backward()

    print("\n--- Sanity Checks ---")
    print(f"Reconstructed patch shape: {recon.shape}")
    print(f"Reconstructed patch dtype: {recon.dtype}")
    print(f"Loss: {loss.item():.4f}")

    # Check if gradients are computed
    first_encoder_param_grad = model.encoder.transformer.layers[0].self_attn.in_proj_weight.grad
    if first_encoder_param_grad is not None:
        print("✅ Gradient computed for an encoder parameter.")
    else:
        print("❌ No gradient found for an encoder parameter.")

    # Check output shape and dtype
    P = model.patch_size
    expected_shape = torch.Size([dummy_input.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    print(f"✅ Output shape is correct: {recon.shape}")

    # The output dtype of autocast can sometimes be float32 even if computations are in bfloat16
    # So we check if it's one of the expected types.
    assert recon.dtype in [torch.bfloat16, torch.float32], f"Dtype mismatch! Got {recon.dtype}"
    print(f"✅ Output dtype is correct: {recon.dtype}")
    print("\nSanity checks passed!")
