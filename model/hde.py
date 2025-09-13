import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Block, VisionTransformer

# --- 1. Spatio-Structural Positional Embedding ---
# A custom module to generate positional embeddings from quadtree data

class SpatioStructuralPosEmbed(nn.Module):
    """
    Generates positional embeddings from patch coordinates and quadtree depths.
    Input:
        - coords: [B, N, 4] tensor of (x1, x2, y1, y2) for each patch.
        - depths: [B, N] tensor of the quadtree depth for each patch.
    Output:
        - [B, N, embed_dim] positional embedding tensor.
    """
    def __init__(self, embed_dim, img_size=224, max_depth=8):
        super().__init__()
        self.img_size = float(img_size)
        self.max_depth = float(max_depth)
        
        # An MLP to project the geometric features into the embedding space
        self.proj = nn.Sequential(
            nn.Linear(5, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, coords, depths):
        # 1. Calculate center (cx, cy), width (w), and height (h) from coordinates
        x1 = coords[..., 0]
        x2 = coords[..., 1]
        y1 = coords[..., 2]
        y2 = coords[..., 3]
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # 2. Normalize geometric features to a [0, 1] range
        cx_norm = cx / self.img_size
        cy_norm = cy / self.img_size
        w_norm = w / self.img_size
        h_norm = h / self.img_size
        depths_norm = depths / self.max_depth
        
        # 3. Concatenate features and project through the MLP
        features = torch.stack([cx_norm, cy_norm, w_norm, h_norm, depths_norm], dim=-1)
        pos_embed = self.proj(features)
        
        return pos_embed

# --- 2. HDE-ViT Main Model ---

class HDEVIT(nn.Module):
    """
    Hierarchical Denoising Encoder (HDE) Vision Transformer.
    Implements MAE-style efficient encoding with dual prediction heads.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=8,
                 decoder_heads=16,
                 norm_pix_loss=True,
                 **kwargs): # Absorb unused kwargs like visible_fraction
        super().__init__()
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # --- 1. Patch Embedding ---
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)

        # --- 2. Spatio-Structural Positional Embedding ---
        self.pos_embed = SpatioStructuralPosEmbed(encoder_dim, img_size)

        # --- 3. Encoder ---
        timm_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels,
            num_classes=0, global_pool='', embed_dim=encoder_dim,
            depth=encoder_depth, num_heads=encoder_heads, mlp_ratio=4.,
            qkv_bias=True, norm_layer=nn.LayerNorm,
        )
        self.encoder_blocks = timm_encoder.blocks
        self.encoder_norm = timm_encoder.norm

        # --- 4. Decoder ---
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_pos_embed = SpatioStructuralPosEmbed(decoder_dim, img_size)
        
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim, num_heads=decoder_heads, mlp_ratio=4.,
                qkv_bias=True, norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # --- 5. Reconstruction Heads (Dual Prediction) ---
        self.head_image = nn.Linear(decoder_dim, patch_dim)
        self.head_noise = nn.Linear(decoder_dim, patch_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, target_patches, pred_patches, target_noise, pred_noise, mask):
        """
        Calculates a combined reconstruction loss for both image and noise
        on the masked (noised) patches.
        """
        target_img = target_patches.flatten(2)
        if self.norm_pix_loss:
            mean = target_img.mean(dim=-1, keepdim=True)
            var = target_img.var(dim=-1, keepdim=True)
            target_img = (target_img - mean) / (var.sqrt() + 1e-6)
        
        loss_img = (pred_patches - target_img) ** 2
        loss_img = loss_img.mean(dim=-1)

        target_n = target_noise.flatten(2)
        mean_n = target_n.mean(dim=-1, keepdim=True)
        var_n = target_n.var(dim=-1, keepdim=True)
        target_n = (target_n - mean_n) / (var_n.sqrt() + 1e-6)
        
        loss_n = (pred_noise - target_n) ** 2
        loss_n = loss_n.mean(dim=-1)

        loss = loss_img + loss_n

        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=target_patches.device)
        loss = (loss * mask).sum() / mask_sum
        return loss

    def forward(self, batch):
        # 1. Unpack data
        patches = batch['patches']
        target_patches = batch['target_patches']
        target_noise = batch['target_noise']
        coords = batch['coords']
        depths = batch['depths']
        mask = batch['mask'] # 1 for noised, 0 for visible

        # 2. Embed all patches and add positional embeddings
        x = patches.flatten(2)
        x = self.patch_embed(x)
        x = x + self.pos_embed(coords, depths)

        # --- 3. MAE-style Efficient Encoding ---
        # Separate visible tokens for the encoder
        # The mask is 0 for visible, so we use `~mask.bool()`
        visible_mask = ~mask.bool()
        x_visible = x[visible_mask].reshape(x.shape[0], -1, x.shape[-1])
        
        # Pass only visible tokens through the encoder
        encoded_visible = self.encoder_blocks(x_visible)
        encoded_visible = self.encoder_norm(encoded_visible)

        # --- 4. Prepare for Decoder ---
        # Project visible tokens to decoder dimension
        encoded_visible_dec = self.decoder_embed(encoded_visible)
        
        # Create full sequence for decoder: scatter visible tokens and fill the rest with mask_tokens
        B, N, C_dec = x.shape[0], x.shape[1], self.decoder_embed.out_features
        x_dec_full = self.mask_token.repeat(B, N, 1)
        
        # Use the mask to create indices for scattering
        # This is a robust way to place the visible tokens back in their original positions
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
        visible_indices = visible_mask.nonzero() # Get coordinates of visible tokens
        
        # Reshape visible tokens to match the flat indexing
        flat_visible_tokens = encoded_visible_dec.reshape(-1, C_dec)
        
        # Scatter visible tokens into the full sequence tensor
        x_dec_full[visible_indices[:, 0], visible_indices[:, 1], :] = flat_visible_tokens

        # Add decoder-specific positional embedding to the full sequence
        x_dec_full = x_dec_full + self.decoder_pos_embed(coords, depths)

        # --- 5. Pass through Decoder ---
        for blk in self.decoder_blocks:
            x_dec_full = blk(x_dec_full)
        x_dec_full = self.decoder_norm(x_dec_full)

        # --- 6. Dual Prediction ---
        recon_patches_all = self.head_image(x_dec_full)
        recon_noise_all = self.head_noise(x_dec_full)

        # --- 7. Calculate Loss ---
        loss = self.forward_loss(target_patches, recon_patches_all,
                                 target_noise, recon_noise_all,
                                 mask)
        
        # --- 8. Reshape for Visualization ---
        C, P = patches.shape[2], patches.shape[3]
        recon_patches_unflat = recon_patches_all.reshape(B, N, C, P, P)

        return loss, recon_patches_unflat, mask


if __name__ == "__main__":
    from torch.amp import autocast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")

    # --- Model & Batch Configuration ---
    B = 8
    C = 3
    img_size = 224
    patch_size = 16
    N = (img_size // patch_size)**2
    visible_fraction = 0.25

    model = HDEVIT(img_size=img_size, patch_size=patch_size).to(device)
    
    # --- Create a dummy batch mimicking the NEW HDE dataloader structure ---
    dummy_target_patches = torch.randn(B, N, C, patch_size, patch_size, device=device)
    dummy_target_noise = torch.randn(B, N, C, patch_size, patch_size, device=device) * 0.1
    
    # --- FIXED: Create a fixed-ratio mask, mimicking the dataloader's behavior ---
    num_visible = int(N * visible_fraction)
    # Generate a random permutation of indices for each item in the batch
    shuffled_indices = torch.rand(B, N, device=device).argsort(dim=1)
    visible_indices = shuffled_indices[:, :num_visible]
    # Create a mask where 1 = noised, 0 = visible
    dummy_mask = torch.ones(B, N, device=device, dtype=torch.long)
    # Use scatter_ to place 0s at the visible indices, marking them as not noised
    dummy_mask.scatter_(dim=1, index=visible_indices, value=0)
    
    dummy_input_patches = dummy_target_patches.clone()
    masked_indices = dummy_mask.bool()
    dummy_input_patches[masked_indices] += dummy_target_noise[masked_indices]
    
    dummy_coords = torch.randint(0, img_size, (B, N, 4), device=device, dtype=torch.float32)
    dummy_depths = torch.randint(0, 8, (B, N), device=device, dtype=torch.float32)

    dummy_batch = {
        'patches': dummy_input_patches,
        'target_patches': dummy_target_patches,
        'target_noise': dummy_target_noise,
        'coords': dummy_coords,
        'depths': dummy_depths,
        'mask': dummy_mask
    }

    # --- Forward and Backward Pass ---
    with autocast(device_type=device.type, dtype=dtype):
        loss, recon_all, mask_out = model(dummy_batch)

    loss.backward()

    print("\n--- Sanity Checks ---")
    print(f"Reconstructed patches shape (full sequence): {recon_all.shape}")
    print(f"Combined Loss: {loss.item():.4f}")
    
    assert torch.equal(mask_out, dummy_mask), "Output mask does not match input mask."
    print("✅ Output mask matches input.")
    
    # Check gradient flow to both heads
    grad_head_image = model.head_image.weight.grad
    grad_head_noise = model.head_noise.weight.grad
    
    if grad_head_image is not None and grad_head_image.abs().sum() > 0:
        print("✅ Gradient computed for image prediction head.")
    else:
        print("❌ No gradient found for image prediction head.")
        
    if grad_head_noise is not None and grad_head_noise.abs().sum() > 0:
        print("✅ Gradient computed for noise prediction head.")
    else:
        print("❌ No gradient found for noise prediction head.")

    # Check the output shape of the full reconstructed sequence
    P = model.patch_size
    expected_shape = torch.Size([B, N, C, P, P])
    assert recon_all.shape == expected_shape, \
        f"Shape mismatch! Got {recon_all.shape}, expected {expected_shape}"
    print(f"✅ Reconstructed output shape is correct: {recon_all.shape}")
    print("\nSanity checks passed!")

