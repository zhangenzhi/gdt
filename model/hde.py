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
    Uses a timm ViT as the encoder backbone and a standard Transformer decoder.
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
        # Projects the flattened raw patch pixels to the encoder's dimension
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, encoder_dim)

        # --- 2. Spatio-Structural Positional Embedding ---
        self.pos_embed = SpatioStructuralPosEmbed(encoder_dim, img_size)

        # --- 3. Encoder ---
        # Instantiate a timm VisionTransformer to use its blocks and norm layer.
        # This approach is flexible, avoids a hardcoded model string, and correctly
        # uses the specified dimensions. We don't use the ViT's own patch or
        # positional embeddings.
        timm_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            num_classes=0,       # Create a headless model
            global_pool='',      # Ensure no global pooling is applied
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        self.encoder_blocks = timm_encoder.blocks
        self.encoder_norm = timm_encoder.norm

        # --- 4. Decoder ---
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # The decoder also needs positional information
        self.decoder_pos_embed = SpatioStructuralPosEmbed(decoder_dim, img_size)
        
        # --- REVISED: Use a ModuleList of timm's `Block` for the decoder ---
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=decoder_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # --- 5. Reconstruction Head ---
        self.head = nn.Linear(decoder_dim, patch_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, target_patches, pred_patches, mask):
        """
        Calculates reconstruction loss only on the noised/masked patches.
        target_patches: [B, N, C, P, P] - Original, un-normalized patches from dataloader
        pred_patches: [B, N, P*P*C] - Model's predictions for all patches
        mask: [B, N] - Boolean mask, True for noised patches
        """
        target = target_patches.flatten(2) # [B, N, P*P*C]
        
        if self.norm_pix_loss:
            # Normalize pixels of each patch independently
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var.sqrt() + 1e-6)
        
        loss = (pred_patches - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N], loss per patch

        # Sum loss only for noised patches and normalize by the number of noised patches
        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=target.device) # Avoid division by zero
        loss = (loss * mask).sum() / mask_sum
        return loss

    def forward(self, batch):
        # 1. Unpack data from the batch dictionary
        patches = batch['patches']     # [B, N, C, P, P]
        coords = batch['coords']       # [B, N, 4]
        depths = batch['depths']       # [B, N]
        mask = batch['mask']           # [B, N], 1 for noised, 0 for visible

        # 2. Embed patches
        x = patches.flatten(2)         # [B, N, C*P*P]
        x = self.patch_embed(x)        # [B, N, D_enc]

        # 3. Add spatio-structural positional embedding
        x = x + self.pos_embed(coords, depths)

        # 4. Pass through encoder
        x = self.encoder_blocks(x)
        x = self.encoder_norm(x)       # [B, N, D_enc]

        # 5. Prepare input for the decoder
        x_dec = self.decoder_embed(x)  # Project to decoder dimension

        # Replace embeddings of noised patches with the mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x_dec)
        mask_tokens = self.mask_token.expand_as(x_dec)
        x_dec = torch.where(mask_expanded.bool(), mask_tokens, x_dec)
        
        # Add decoder-specific positional embedding
        x_dec = x_dec + self.decoder_pos_embed(coords, depths)

        # 6. Pass through decoder
        # --- REVISED: Manually iterate through the decoder blocks ---
        for blk in self.decoder_blocks:
            x_dec = blk(x_dec)
        x_dec = self.decoder_norm(x_dec) # [B, N, D_dec]

        # 7. Reconstruct all patches
        recon_patches_all = self.head(x_dec) # [B, N, P*P*C]

        # 8. Calculate loss on noised patches
        loss = self.forward_loss(patches, recon_patches_all, mask)
        
        # --- FIXED: Return the full reconstructed tensor. ---
        # The training loop's visualizer will use the mask to select the correct patches.
        # This is a more robust approach than trying to filter inside the model.
        
        # Reshape to [B, N, C, P, P] for visualization function
        B, N, _ = recon_patches_all.shape
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
    
    # --- Create a dummy batch mimicking the HDE dataloader ---
    dummy_patches = torch.randn(B, N, C, patch_size, patch_size, device=device)
    dummy_coords = torch.randint(0, img_size, (B, N, 4), device=device, dtype=torch.float32)
    dummy_depths = torch.randint(0, 8, (B, N), device=device, dtype=torch.float32)
    
    # Create a mask where 1 = noised, 0 = visible
    dummy_mask = (torch.rand(B, N, device=device) > visible_fraction).long()
    
    dummy_batch = {
        'patches': dummy_patches,
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
    print(f"Reconstructed patches dtype: {recon_all.dtype}")
    print(f"Loss: {loss.item():.4f}")
    
    # Check if the output mask matches the input mask
    assert torch.equal(mask_out, dummy_mask), "Output mask does not match input mask."
    print("✅ Output mask matches input.")
    
    # Check the gradient flow to the encoder
    first_encoder_param_grad = model.encoder_blocks[0].attn.qkv.weight.grad
    if first_encoder_param_grad is not None and first_encoder_param_grad.abs().sum() > 0:
        print("✅ Gradient computed for an encoder parameter.")
    else:
        print("❌ No gradient found for an encoder parameter.")

    # Check the gradient flow to the new decoder
    first_decoder_param_grad = model.decoder_blocks[0].attn.qkv.weight.grad
    if first_decoder_param_grad is not None and first_decoder_param_grad.abs().sum() > 0:
        print("✅ Gradient computed for a decoder parameter.")
    else:
        print("❌ No gradient found for a decoder parameter.")

    # Check the output shape of the full reconstructed sequence
    P = model.patch_size
    expected_shape = torch.Size([B, N, C, P, P])
    assert recon_all.shape == expected_shape, \
        f"Shape mismatch! Got {recon_all.shape}, expected {expected_shape}"

    print(f"✅ Reconstructed output shape is correct: {recon_all.shape}")

    # Use the mask to get the reconstructed *noised* patches for a downstream task (like visualization)
    recon_noised = recon_all[mask_out.bool()]
    total_masked_patches = dummy_mask.sum().item()
    # The shape will be [Total_Masked_in_Batch, C, P, P]
    assert recon_noised.shape[0] == total_masked_patches
    print(f"✅ Correctly filtered {recon_noised.shape[0]} noised patches.")

    assert recon_all.dtype in [torch.bfloat16, torch.float32], f"Dtype mismatch! Got {recon_all.dtype}"
    print(f"✅ Reconstructed output dtype is correct: {recon_all.dtype}")
    print("\nSanity checks passed!")

