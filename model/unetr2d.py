import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import PatchEmbed
from timm.layers import resample_patch_embed, PatchEmbed, Mlp, DropPath, lecun_normal_, trunc_normal_
import logging
from typing import Tuple, Optional, List

_logger = logging.getLogger(__name__)

# --- Helper Decoder Blocks ---

class UNETRDecoderBlock(nn.Module):
    """
    A single decoder block for UNETR, combining skip connection and upsampling.
    Uses ConvTranspose2d for upsampling.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Upsampling layer (doubles spatial resolution)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

        # Convolution block after concatenation
        self.conv = nn.Sequential(
            # Corrected input channels calculation
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_below: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_below (torch.Tensor): Feature map from the layer below in the decoder. NCHW format.
            x_skip (torch.Tensor): Feature map from the corresponding encoder layer (skip connection). NCHW format.
        """
        x_up = self.upsample(x_below)

        # Ensure spatial dimensions match before concatenation
        if x_up.shape[2:] != x_skip.shape[2:]:
             # Pad or interpolate x_up if needed
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)
            _logger.debug(f"Interpolated x_up to match x_skip size: {x_skip.shape[2:]}")


        x_concat = torch.cat([x_up, x_skip], dim=1)
        x_out = self.conv(x_concat)
        return x_out


# --- Positional Embedding Interpolation ---
# From timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
def resample_abs_pos_embed(
        posemb,
        new_grid_size: List[int],
        old_grid_size: List[int],
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample absolute position embedding grid B x (N + T) x C -> B x (M + T) x C"""
    T = num_prefix_tokens
    posemb_prefix = posemb[:, :T] if T else torch.empty(posemb.shape[0], 0, posemb.shape[2])
    posemb_grid = posemb[:, T:]

    # Assuming B x N x C layout
    if new_grid_size[0] == old_grid_size[0] and new_grid_size[1] == old_grid_size[1]:
        return posemb # No need to resample if grid sizes match

    if verbose:
        _logger.info(f'Resampling pos_embed, grid {old_grid_size} -> {new_grid_size}, T={T}')

    # Reshape B x N x C -> B x C x H x W
    posemb_grid = posemb_grid.transpose(-1, -2).reshape(posemb.shape[0], posemb.shape[2], old_grid_size[0], old_grid_size[1])

    posemb_grid = F.interpolate(
        posemb_grid,
        size=new_grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )

    # Reshape back to B x N x C
    posemb_grid = posemb_grid.reshape(posemb.shape[0], posemb.shape[2], -1).transpose(-1, -2)

    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


class UNETR2D(nn.Module):
    """
    2D UNETR model using a timm ViT backbone.
    Handles BNC format internally for ViT blocks.
    """
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224',
        img_size: int = 1024,
        in_chans: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
        feature_indices: Tuple[int, ...] = (2, 5, 8, 11),
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
        # Add parameter for pos embed interpolation mode
        pos_embed_interp: str = 'bicubic',
        pos_embed_antialias: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_indices = feature_indices
        self.img_size = img_size
        self.in_chans = in_chans
        self.pos_embed_interp = pos_embed_interp
        self.pos_embed_antialias = pos_embed_antialias

        # --- Encoder (ViT Backbone) ---
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            # img_size=img_size, # Timm ViT doesn't always use img_size directly here
            # Request features from specific blocks AND the final output
            features_only=False, # We need access to blocks manually
        )

        # Store original pos embed and grid size
        self.orig_pos_embed = self.backbone.pos_embed.data.clone() if self.backbone.pos_embed is not None else None
        # Infer original grid size from pos_embed (excluding class token if present)
        num_pos_tokens = self.backbone.pos_embed.shape[1] if self.backbone.pos_embed is not None else 0
        num_prefix_tokens = self.backbone.num_prefix_tokens
        num_patch_tokens = num_pos_tokens - num_prefix_tokens
        self.orig_grid_size = int(num_patch_tokens ** 0.5)
        self.num_prefix_tokens = num_prefix_tokens # Store number of prefix tokens (e.g., class token)

        # Check patch embed config and store grid size calculation function
        self.grid_size = self.backbone.patch_embed.grid_size
        encoder_embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]

        # --- Manually adjust patch embedding if input channels don't match ---
        if in_chans != self.backbone.patch_embed.proj.in_channels:
             _logger.warning(
                 f"Input channels ({in_chans}) mismatch pretrained ({self.backbone.patch_embed.proj.in_channels}). "
                 f"Adapting patch embedding projection."
             )
             orig_weight = self.backbone.patch_embed.proj.weight.data
             orig_bias = self.backbone.patch_embed.proj.bias.data if self.backbone.patch_embed.proj.bias is not None else None
             new_proj = nn.Conv2d(
                 in_chans,
                 encoder_embed_dim,
                 kernel_size=self.backbone.patch_embed.patch_size,
                 stride=self.backbone.patch_embed.stride,
                 padding=self.backbone.patch_embed.padding
             )

             # Adapt weights (simple averaging if going from 3 to 1)
             if in_chans == 1 and orig_weight.shape[1] == 3:
                 new_proj.weight.data = orig_weight.mean(dim=1, keepdim=True)
             elif in_chans==3 and orig_weight.shape[1]==1:
                 # If pretrained was 1 channel and we need 3, replicate
                  _logger.warning("Replicating single-channel patch embed weights to 3 channels.")
                  new_proj.weight.data = orig_weight.repeat(1, 3, 1, 1)
             else:
                 _logger.warning("Could not automatically adapt patch embedding weights for channels. Using random init for proj.")
                 # Keep random init

             if orig_bias is not None:
                 new_proj.bias.data = orig_bias
             self.backbone.patch_embed.proj = new_proj

        # --- Skip Connection Feature Dimensions ---
        # For standard ViT, all intermediate blocks have the same embed_dim
        skip_channel_dims = [encoder_embed_dim] * len(feature_indices)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Decoder input starts from the deepest selected feature
        decoder_in_channels = skip_channel_dims[-1]

        # Build decoder blocks from deep to shallow
        # Ensure enough decoder channels are provided
        if len(decoder_channels) != len(feature_indices):
            raise ValueError(f"Length of decoder_channels ({len(decoder_channels)}) must match length of feature_indices ({len(feature_indices)})")

        for i in range(len(feature_indices) - 1, 0, -1):
            skip_ch = skip_channel_dims[i-1]
            out_ch = decoder_channels[i] # Output channels for this decoder stage
            self.decoder_blocks.append(
                UNETRDecoderBlock(decoder_in_channels, skip_ch, out_ch)
            )
            decoder_in_channels = out_ch # Input for the next block is the output of this one

        # --- Final Blocks ---
        # Final block combines output of the last standard decoder block with the shallowest skip connection
        self.final_block = UNETRDecoderBlock(
            decoder_in_channels, skip_channel_dims[0], decoder_channels[0]
        )

        # Additional upsampling needed to reach original image size
        # Upsample from the output of final_block (resolution H/P * 2) to H
        final_up_factor = self.patch_size // 2
        num_final_upsamples = int(torch.log2(torch.tensor(float(final_up_factor))).item()) if final_up_factor > 1 else 0

        upsample_layers = []
        current_channels = decoder_channels[0]
        for _ in range(num_final_upsamples):
            next_channels = max(16, current_channels // 2) # Halve channels, min 16
            upsample_layers.append(nn.ConvTranspose2d(current_channels, next_channels, kernel_size=2, stride=2))
            upsample_layers.append(nn.BatchNorm2d(next_channels))
            upsample_layers.append(nn.ReLU(inplace=True))
            current_channels = next_channels

        self.final_upsample = nn.Sequential(*upsample_layers)

        # Final 1x1 convolution
        self.segmentation_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def _interpolate_pos_embed(self, x, H_grid: int, W_grid: int):
        """ Handles positional embedding interpolation. """
        if self.orig_pos_embed is None:
            return x # No pos embed to interpolate

        # target shape for interpolation (excluding prefix tokens)
        new_grid_size = (H_grid, W_grid)
        old_grid_size = (self.orig_grid_size, self.orig_grid_size)

        if new_grid_size != old_grid_size:
            #_logger.info(f"Interpolating pos embed from {old_grid_size} to {new_grid_size}")
            pos_embed_interpolated = resample_abs_pos_embed(
                self.orig_pos_embed,
                new_grid_size=list(new_grid_size),
                old_grid_size=list(old_grid_size),
                num_prefix_tokens=self.num_prefix_tokens,
                interpolation=self.pos_embed_interp,
                antialias=self.pos_embed_antialias,
                verbose=False, # Set to True for debugging
            )
            # Add interpolated pos embed to x (which is BNC)
            x = x + pos_embed_interpolated[:, self.num_prefix_tokens:, :]
        else:
            # Add original pos embed
            x = x + self.orig_pos_embed[:, self.num_prefix_tokens:, :]

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder Forward ---
        # 1. Patch Embedding
        x_patch = self.backbone.patch_embed(x) # Output BNC: [B, N, C]
        B, N, C = x_patch.shape
        H_grid, W_grid = self.backbone.patch_embed.grid_size # Get grid size

        # Handle prefix tokens (e.g., class token) if they exist
        if self.num_prefix_tokens > 0:
            cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
            x_patch = torch.cat((cls_tokens, x_patch), dim=1)

        # 2. Add Positional Embedding (Interpolated)
        x_with_pos = self._interpolate_pos_embed(x_patch, H_grid, W_grid)
        x_with_pos = self.backbone.pos_drop(x_with_pos)

        # Remove prefix tokens before passing to blocks if blocks don't expect them?
        # Standard ViT blocks expect the full sequence including class token.
        current_x_bnc = x_with_pos # Keep BNC format for blocks

        skip_connections = {} # Store skips by index

        # 3. Iterate through ViT Blocks
        for i, blk in enumerate(self.backbone.blocks):
            current_x_bnc = blk(current_x_bnc)
            # Store output if it's one of the feature indices
            if i in self.feature_indices:
                # Remove prefix tokens before reshaping? Assume skip features don't include cls token.
                feature_bnc = current_x_bnc[:, self.num_prefix_tokens:, :]
                # Reshape BNC -> B, H, W, C -> B, C, H, W for decoder
                feature_nchw = feature_bnc.reshape(B, H_grid, W_grid, C).permute(0, 3, 1, 2).contiguous()
                skip_connections[i] = feature_nchw
                # _logger.debug(f"Stored skip connection from block {i}: {feature_nchw.shape}")


        # --- Decoder Forward ---
        # Ensure skip connections were extracted correctly
        if len(skip_connections) != len(self.feature_indices):
             raise RuntimeError(f"Extracted {len(skip_connections)} skip connections, but expected {len(self.feature_indices)}. Check feature_indices.")

        # Sort indices to process from deep to shallow
        sorted_indices = sorted(self.feature_indices, reverse=True)

        # Start with the deepest feature map
        decoder_x = skip_connections[sorted_indices[0]]
        # _logger.debug(f"Decoder input start (deepest skip {sorted_indices[0]}): {decoder_x.shape}")


        # Apply decoder blocks, combining with skip connections from shallower layers
        for i in range(len(self.decoder_blocks)):
            # Skip connection comes from the next shallower feature index
            skip_index = sorted_indices[i + 1]
            skip = skip_connections[skip_index]
            # _logger.debug(f"Applying decoder block {i} with skip {skip_index} ({skip.shape}) to input ({decoder_x.shape})")
            decoder_x = self.decoder_blocks[i](decoder_x, skip)
            # _logger.debug(f"Output of decoder block {i}: {decoder_x.shape}")


        # Apply final block with the shallowest skip connection
        # Skip connection comes from the shallowest feature index
        # skip_index_shallowest = sorted_indices[-1] # This was already used in the loop
        # Instead, the final block uses the output of the last decoder_block and the shallowest skip
        skip_shallowest = skip_connections[sorted_indices[-1]] # Shallowest stored skip
        # _logger.debug(f"Applying final block with shallowest skip {sorted_indices[-1]} ({skip_shallowest.shape}) to input ({decoder_x.shape})")
        # --- Correction: Final block input should be output of last loop iter ---
        # The loop iterates len(decoder_blocks) times = len(feature_indices) - 1
        # The final skip connection needed is feature_indices[0] (shallowest)
        skip_0 = skip_connections[self.feature_indices[0]] # Get shallowest skip by original index
        decoder_x = self.final_block(decoder_x, skip_0)
        # _logger.debug(f"Output of final block: {decoder_x.shape}")


        # --- Final Upsampling & Output ---
        x_out = self.final_upsample(decoder_x)
        # _logger.debug(f"Output after final upsample: {x_out.shape}")

        logits = self.segmentation_head(x_out)
        # _logger.debug(f"Output after segmentation head: {logits.shape}")


        # Ensure final output size matches input size
        if logits.shape[2:] != x.shape[2:]:
            # _logger.debug(f"Interpolating final output from {logits.shape[2:]} to {x.shape[2:]}")
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits

# --- Factory Function ---
def create_unetr_model(
    backbone_name: str = 'vit_base_patch16_224',
    img_size: int = 1024,
    in_chans: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    feature_indices: Tuple[int, ...] = (2, 5, 8, 11), # Default for ViT-B (depth 12)
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32), # Example decoder channels
) -> UNETR2D:
    """
    Creates a UNETR-2D model with a specified timm ViT backbone.
    """
    _logger.info(
        f"Creating UNETR-2D model with backbone: {backbone_name}, img_size: {img_size}, "
        f"in_chans: {in_chans}, num_classes: {num_classes}"
    )
    # Validate feature indices against known backbone depth (simple check)
    try:
        # Attempt to get default config to infer depth, might fail for some models
        default_cfg = timm.get_pretrained_cfg(backbone_name)
        backbone_depth = default_cfg.get('depth', None) if default_cfg else None
        if backbone_depth and max(feature_indices) >= backbone_depth:
            _logger.warning(
                f"Max feature index ({max(feature_indices)}) >= backbone depth ({backbone_depth}). "
                f"Ensure indices are valid for '{backbone_name}'."
            )
    except Exception:
        _logger.warning(f"Could not automatically determine depth for '{backbone_name}'. Please verify feature_indices.")

    model = UNETR2D(
        backbone_name=backbone_name,
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=pretrained,
        feature_indices=feature_indices,
        decoder_channels=decoder_channels,
    )
    return model

# --- Local Test ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Use INFO for cleaner output

    img_size = 1024
    in_chans = 1
    num_classes = 1

    # Using ViT-Base with patch size 16
    model = create_unetr_model(
        backbone_name='vit_base_patch16_224', # Standard ViT-B backbone
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=True, # Load pretrained weights for the backbone
        feature_indices=(2, 5, 8, 11), # Indices for ViT-Base (depth 12)
        decoder_channels=(256, 128, 64, 32), # Example channel configuration
    )

    _logger.info(f"UNETR Model created successfully.")

    # Test forward pass
    dummy_input = torch.randn(2, in_chans, img_size, img_size) # Batch size 2
    _logger.info(f"Input shape: {dummy_input.shape}")

    try:
        model.eval() # Set model to evaluation mode for testing
        with torch.no_grad():
            output = model(dummy_input)
        _logger.info(f"Output shape: {output.shape}")
        assert output.shape == (2, num_classes, img_size, img_size), "Output shape mismatch!"
        _logger.info("Forward pass successful and output shape is correct.")
    except Exception as e:
        _logger.error(f"Error during forward pass: {e}", exc_info=True)
        # Print model structure might be too verbose, rely on traceback

