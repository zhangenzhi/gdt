# --------------------------------------------- #
#   File: mae.py
#   MAE Model Definition (Revised and Annotated)
# --------------------------------------------- #
import torch
import torch.nn as nn
from torch.amp import autocast

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
#   2️⃣  Encoder (Refactored with CLS token)
# --------------------------------------------- #
class MAEEncoder(nn.Module):
    """Encoder that processes only visible patches with a CLS token."""
    def __init__(self, *, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # REVISED: Add CLS token and update positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
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

    # REVISED: Forward pass simplified to work with pre-selected tokens
    def forward(self, x: torch.Tensor, ids_keep: torch.Tensor):
        # CORRECTED: Get batch size from 4D tensor
        B = x.shape[0]

        # Patchify and add positional embeddings
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :] # Add pos embed to patches

        # Select only the visible patches based on ids_keep
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # Prepend CLS token and its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        
        # Pass through transformer encoder
        # No padding mask needed as all sequences are of the same length
        enc_out = self.transformer(x)
        enc_out = self.norm(enc_out)
        return enc_out

# --------------------------------------------- #
#   3️⃣  Decoder (Refactored for index-based scatter)
# --------------------------------------------- #
class MAEDecoder(nn.Module):
    """Decoder that reconstructs all patches using an index-based scatter."""
    def __init__(self, *, encoder_dim=768, decoder_dim=512, decoder_depth=4,
                 decoder_heads=16, patch_size=16, img_size=224):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_dim))

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

    # REVISED: Forward pass uses indices to reconstruct the full sequence
    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # Embed encoder tokens to the decoder's dimension
        x = self.decoder_embed(x)
        
        # Separate CLS token from patch tokens
        cls_token = x[:, :1, :]
        patch_tokens = x[:, 1:, :]

        # Create a placeholder for the full sequence of patches, filled with mask tokens
        B = x.shape[0]
        full_sequence = self.mask_token.repeat(B, self.num_patches, 1)

        # Use scatter to place the visible patch tokens back into their original positions
        # ids_restore is used to "un-shuffle" the sequence
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        full_sequence.scatter_(dim=1, index=ids_restore_expanded, src=patch_tokens)
        
        # Add positional embeddings to the full sequence of patches
        full_sequence = full_sequence + self.pos_embed[:, 1:, :]

        # Pass the full sequence through the decoder
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)

        # Project to pixel space for reconstruction
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat


# --------------------------------------------- #
#   4️⃣  MAE wrapper (Refactored masking and data flow)
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
                 decoder_dim=512,
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
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=encoder_dim, depth=encoder_depth, num_heads=encoder_heads,
        )
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim, decoder_dim=decoder_dim,
            decoder_depth=decoder_depth, decoder_heads=decoder_heads,
            patch_size=patch_size, img_size=img_size,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0, "Image dimensions must be divisible by patch size."
        num_patches_h = H // p
        x = imgs.reshape(B, C, num_patches_h, p, num_patches_h, p)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B, self.num_patches, C * p * p)
        return x

    # REVISED: Fixed-ratio masking using randperm
    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        # Generate noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise and keep the top `len_keep` indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        
        # Generate the boolean mask (True for masked)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Un-shuffle the mask to its original order
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask.bool(), ids_restore

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    # REVISED: Main forward pass uses index-based logic
    def forward(self, x: torch.Tensor):
        # Generate fixed-ratio mask and corresponding indices
        # We patchify the image inside the encoder now
        _, ids_keep, mask, ids_restore = self.random_masking(
            torch.zeros(x.shape[0], self.num_patches, 1, device=x.device)
        )
        
        # Encode only the visible patches
        enc_out = self.encoder(x, ids_keep)

        # Decode the full sequence of patches
        recon_patches_flat = self.decoder(enc_out, ids_restore)

        # Calculate the reconstruction loss
        loss = self.forward_loss(x, recon_patches_flat, mask)

        return loss, recon_patches_flat, mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")

    model = MAE().to(device)
    dummy_input = torch.randn(8, 3, 224, 224, device=device)

    with autocast(device.type, dtype=dtype):
        loss, recon, mask = model(dummy_input)

    loss.backward()

    print("\n--- Sanity Checks ---")
    print(f"Reconstructed patch shape: {recon.shape}")
    print(f"Reconstructed patch dtype: {recon.dtype}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Number of masked patches: {mask.sum().item()}")
    print(f"Expected number of masked patches: {int(model.num_patches * model.mask_ratio) * dummy_input.shape[0]}")


    first_encoder_param_grad = model.encoder.transformer.layers[0].self_attn.in_proj_weight.grad
    if first_encoder_param_grad is not None:
        print("✅ Gradient computed for an encoder parameter.")
    else:
        print("❌ No gradient found for an encoder parameter.")

    P = model.patch_size
    expected_shape = torch.Size([dummy_input.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    print(f"✅ Output shape is correct: {recon.shape}")

    assert recon.dtype in [torch.bfloat16, torch.float32], f"Dtype mismatch! Got {recon.dtype}"
    print(f"✅ Output dtype is correct: {recon.dtype}")
    print("\nSanity checks passed!")
