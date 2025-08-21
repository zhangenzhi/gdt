# --------------------------------------------- #
#   File: mae.py
#   MAE Model Definition
# --------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast

# --------------------------------------------- #
#   1️⃣  Patch embedding
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
#   2️⃣  Encoder (Efficient Version)
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
        visible_patches_list = [x[i][mask[i]] for i in range(B)]
        padded_visible_patches = pad_sequence(visible_patches_list, batch_first=True)
        num_visible = mask.sum(dim=1)
        max_vis = num_visible.max()
        padding_mask = torch.arange(max_vis, device=x.device)[None, :] >= num_visible[:, None]
        enc_out = self.transformer(padded_visible_patches, src_key_padding_mask=padding_mask)
        enc_out = self.norm(enc_out)
        return enc_out, visible_patches_list

# --------------------------------------------- #
#   3️⃣  Decoder (Efficient Version)
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
        full_sequence = self.mask_token.to(visible_tokens.dtype).repeat(B, N, 1)
        full_sequence[mask] = flat_unpadded_tokens
        full_sequence = full_sequence + self.pos_embed.to(visible_tokens.dtype)
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat

# --------------------------------------------- #
#   4️⃣  MAE wrapper (handles masking)
# --------------------------------------------- #
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
                 mask_ratio=0.75,
                 norm_pix_loss=True):  # Added new parameter
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss # Store the parameter

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
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_size
        N, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = w = H // p
        x = imgs.reshape(shape=(N, C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(N, h * w, C * p**2))
        return x

    def random_masking(self, B: int, device: torch.device) -> torch.BoolTensor:
        """
        Random binary mask – True = visible patch
        """
        return torch.rand(B, self.num_patches, device=device) < (1.0 - self.mask_ratio)

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * (~mask)).sum() / (~mask).sum()  # mean loss on masked patches
        return loss

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
        enc_out, _ = self.encoder(x, mask)
        
        # Decode full sequence
        recon_patches_flat = self.decoder(enc_out, mask) # [B, N, P*P*3]
        
        # Calculate loss on masked patches using the new function
        loss = self.forward_loss(x, recon_patches_flat, mask)
        
        return loss, recon_patches_flat, mask


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

    # Add backward pass to verify gradients
    loss.backward()
    
    print("Reconstructed patch shape:", recon.shape)
    print("Reconstructed patch dtype:", recon.dtype)
    print("Loss:", loss.item())
    
    # Check if a gradient exists for a parameter in the encoder
    # This is a good way to test if the backward pass worked.
    first_encoder_param_grad = model.encoder.transformer.layers[0].self_attn.in_proj_weight.grad
    if first_encoder_param_grad is not None:
        print("Gradient for the first encoder layer's attention weight has been calculated.")
    else:
        print("No gradient was calculated for the first encoder layer's attention weight.")
        
    P = model.patch_size
    expected_shape = torch.Size([dummy.size(0), model.num_patches, 3 * P * P])
    assert recon.shape == expected_shape, f"Shape mismatch! Got {recon.shape}, expected {expected_shape}"
    assert recon.dtype == torch.bfloat16, f"Dtype mismatch! Got {recon.dtype}, expected {torch.bfloat16}"
    print("Sanity check passed!")