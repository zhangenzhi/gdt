# --------------------------------------------- #
#   File: mae.py
#   MAE Model Definition (Using timm for Encoder)
# --------------------------------------------- #
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed, Block

# --------------------------------------------- #
#   1️⃣  Encoder (Refactored to use timm's ViT)
# --------------------------------------------- #
class MAEEncoder(nn.Module):
    """
    MAE Encoder implemented using a Vision Transformer from the timm library.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=False, **kwargs):
        super().__init__()
        # Create a ViT model from timm
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # We will manually handle the forward pass to implement the MAE logic,
        # so we only need the components from the timm model.
        # The following attributes are aliased for clarity and direct access,
        # matching the original MAE paper's variable names where possible.
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.patch_embed = self.model.patch_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm

    def forward(self, x: torch.Tensor, ids_keep: torch.Tensor):
        B = x.shape[0]

        # 1. Patchify and add positional embeddings
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Add pos embed to patch tokens

        # 2. Select only the visible patches based on ids_keep
        # This is the core of the MAE encoder logic
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        # 3. Prepend the CLS token and its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. Pass the reduced sequence of tokens through the transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # 5. Apply the final normalization
        enc_out = self.norm(x)
        return enc_out

# --------------------------------------------- #
#   2️⃣  Decoder (No changes needed)
# --------------------------------------------- #
class MAEDecoder(nn.Module):
    """Decoder that reconstructs all patches using an index-based gather."""
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

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(x)
        num_masked = self.num_patches - (x.shape[1] - 1)
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        x_shuffled = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_restored = torch.gather(x_shuffled, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        full_sequence = x_restored + self.pos_embed
        dec_out = self.decoder(full_sequence)
        dec_out = self.norm(dec_out)
        recon_flat = self.reconstruction_head(dec_out)
        return recon_flat

# --------------------------------------------- #
#   3️⃣  MAE wrapper (Minor change in __init__)
# --------------------------------------------- #
class MAE(nn.Module):
    """End-to-end Masked Auto-Encoder."""
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_dim=768, # Note: This is now implicitly defined by the timm model
                 decoder_dim=512,
                 decoder_depth=4,
                 decoder_heads=16,
                 mask_ratio=0.75,
                 norm_pix_loss=True,
                 **kwargs): # Absorb other encoder args
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss

        # Use the new timm-based encoder
        self.encoder = MAEEncoder(model_name='vit_base_patch16_224')
        
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_size=patch_size,
            img_size=img_size,
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

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
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

    def forward(self, x: torch.Tensor):
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