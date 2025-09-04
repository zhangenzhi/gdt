import torch
import torch.nn as nn
import timm
# Import the specific classes from timm for direct instantiation
from timm.models.vision_transformer import VisionTransformer, Block

# --------------------------------------------------------------------------
# MAE Encoder, Decoder, and Model classes
# --------------------------------------------------------------------------

class MAEEncoder(nn.Module):
    """
    MAE Encoder implemented by directly instantiating the VisionTransformer 
    class from `timm` based on specified image and patch sizes.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        # Directly instantiate the VisionTransformer class from timm.
        # Setting num_classes=0 makes the model a headless feature extractor,
        # which is perfect for our needs and avoids DDP issues.
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,       # Create a headless model
            global_pool='',      # Ensure no global pooling is applied
            mlp_ratio=4.,        # Standard MLP ratio
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
        )
        
        # Alias essential components from the ViT model for clarity
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.patch_embed = self.model.patch_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm

    def forward(self, x: torch.Tensor, ids_keep: torch.Tensor):
        B = x.shape[0]

        # 1. Patchify and add positional embeddings to patch tokens
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # 2. Select only the visible patches (the core MAE logic)
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # 3. Prepend the CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Pass the reduced sequence through the transformer blocks
        x = self.blocks(x) # timm's ModuleList of Blocks can be called directly
        
        # 5. Apply final normalization
        enc_out = self.norm(x)
        return enc_out

class MAEDecoder(nn.Module):
    """
    MAE Decoder, revised to use timm's `Block` for its transformer layers.
    """
    def __init__(self, *, img_size=224, patch_size=16, in_chans=3,
                 encoder_dim=768, decoder_dim=512, decoder_depth=8,
                 decoder_heads=16):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))

        # --- CRITICAL REVISION ---
        # Use a ModuleList of timm's `Block` instead of torch.nn.TransformerEncoder
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim,
                num_heads=decoder_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            )
            for _ in range(decoder_depth)])
        
        # The output dimensions must match the input channels (e.g., 1 for grayscale)
        self.reconstruction_head = nn.Linear(decoder_dim, patch_size**2 * in_chans, bias=True)
        
        self.norm = nn.LayerNorm(decoder_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)
        num_masked = self.num_patches - (x.shape[1] - 1)
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        
        # Combine visible patch tokens with mask tokens
        x_shuffled = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        # Restore original patch order
        x_restored = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # Add positional embeddings to the full sequence
        full_sequence = x_restored + self.pos_embed
        
        # Pass the sequence through the decoder blocks
        for blk in self.decoder_blocks:
            full_sequence = blk(full_sequence)
            
        dec_out = self.norm(full_sequence)
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat

class MAE(nn.Module):
    """
    End-to-end Masked Auto-Encoder, fully configurable for custom datasets.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=8,
                 decoder_heads=16,
                 mask_ratio=0.75,
                 norm_pix_loss=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss

        # Instantiate encoder and decoder with the specified configurations
        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_dim, depth=encoder_depth, num_heads=encoder_heads
        )
        
        self.decoder = MAEDecoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            encoder_dim=encoder_dim, decoder_dim=decoder_dim,
            decoder_depth=decoder_depth, decoder_heads=decoder_heads
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0, "Image dimensions must be square and divisible by patch size."
        num_patches_h = H // p
        x = imgs.reshape(B, C, num_patches_h, p, num_patches_h, p)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B, self.num_patches, C * p * p)
        return x

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Generate the binary mask (1 for masked, 0 for visible)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return ids_keep, mask.bool(), ids_restore

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / ((var + 1.e-6)**.5)
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # Loss per patch
        loss = (loss * mask).sum() / mask.sum() # Average loss on masked patches
        return loss

    def forward(self, x: torch.Tensor):
        # The input to random_masking is just a placeholder to get shapes right
        ids_keep, mask, ids_restore = self.random_masking(
            torch.zeros(x.shape[0], self.num_patches, 1, device=x.device)
        )
        
        enc_out = self.encoder(x, ids_keep)
        recon_patches_flat = self.decoder(enc_out, ids_restore)
        loss = self.forward_loss(x, recon_patches_flat, mask)
        
        return loss, recon_patches_flat, mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Testing MAE with timm Encoder ---")
    model = MAE().to(device)
    
    # Verify that the encoder is indeed a timm model
    print(f"Encoder class: {type(model.encoder.model)}")
    
    dummy_input = torch.randn(8, 3, 224, 224, device=device)

    with torch.amp.autocast(device.type, dtype=torch.bfloat16):
        loss, recon, mask = model(dummy_input)

    loss.backward()

    print("\n--- Sanity Checks ---")
    print(f"Reconstructed patch shape: {recon.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    first_encoder_param_grad = model.encoder.model.blocks[0].attn.qkv.weight.grad
    if first_encoder_param_grad is not None:
        print("✅ Gradient computed for a timm encoder parameter.")
    else:
        print("❌ No gradient found for a timm encoder parameter.")
    print("\nSanity checks passed!")
