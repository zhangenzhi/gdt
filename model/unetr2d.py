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
    # Corrected reshape based on original grid size
    if posemb_grid.shape[1] != old_grid_size[0] * old_grid_size[1]:
        raise ValueError(f"posemb_grid size {posemb_grid.shape[1]} does not match old_grid_size {old_grid_size}")
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
            img_size=img_size, # *** Explicitly set img_size for backbone ***
            features_only=False, # We need access to blocks manually
        )

        # Store original pos embed and grid size
        self.orig_pos_embed = self.backbone.pos_embed.data.clone() if self.backbone.pos_embed is not None else None
        # Infer original grid size from pos_embed (excluding class token if present)
        # Handle cases where pos_embed might be None or not have expected shape
        if self.backbone.pos_embed is not None and len(self.backbone.pos_embed.shape) == 3 and self.backbone.pos_embed.shape[1] > self.backbone.num_prefix_tokens:
            num_pos_tokens = self.backbone.pos_embed.shape[1]
            num_prefix_tokens = self.backbone.num_prefix_tokens
            num_patch_tokens = num_pos_tokens - num_prefix_tokens
            # Ensure num_patch_tokens is a perfect square
            if int(num_patch_tokens ** 0.5) ** 2 == num_patch_tokens:
                self.orig_grid_size = int(num_patch_tokens ** 0.5)
            else:
                _logger.warning(f"Could not infer original grid size from pos_embed shape {self.backbone.pos_embed.shape} and prefix tokens {num_prefix_tokens}. Using patch_embed grid size.")
                # Fallback to patch_embed grid_size if inference fails
                self.orig_grid_size = self.backbone.patch_embed.grid_size[0] # Assume square
        else:
             _logger.warning(f"Pos_embed not found or has unexpected shape ({self.backbone.pos_embed.shape if self.backbone.pos_embed is not None else 'None'}). Using patch_embed grid size for original.")
             self.orig_grid_size = self.backbone.patch_embed.grid_size[0] # Assume square

        self.num_prefix_tokens = self.backbone.num_prefix_tokens # Store number of prefix tokens (e.g., class token)


        # Check patch embed config and store grid size calculation function
        self.grid_size = self.backbone.patch_embed.grid_size
        encoder_embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.patch_size[0]

        # --- Manually adjust patch embedding if input channels don't match ---
        # Note: timm's create_model with in_chans often handles this, but double-checking
        if in_chans != self.backbone.patch_embed.proj.in_channels:
             _logger.warning(
                 f"Input channels ({in_chans}) potentially mismatch pretrained backbone ({self.backbone.patch_embed.proj.in_channels}). "
                 f"Timm's create_model usually adapts this, but verify weights if issues arise."
             )
             # The adaptation logic previously here might be redundant if timm handles it.
             # Keeping the logic just in case, but it might need refinement based on how timm truly handles adaptation.
             if self.backbone.patch_embed.proj.weight.shape[1] == 3 and in_chans == 1:
                  _logger.info("Attempting manual adaptation of patch embed from 3 to 1 channels (average).")
                  orig_weight = self.backbone.patch_embed.proj.weight.data
                  orig_bias = self.backbone.patch_embed.proj.bias.data if self.backbone.patch_embed.proj.bias is not None else None
                  new_proj = nn.Conv2d(
                      in_chans, encoder_embed_dim, kernel_size=self.backbone.patch_embed.patch_size,
                      stride=self.backbone.patch_embed.stride, padding=self.backbone.patch_embed.padding
                  )
                  new_proj.weight.data = orig_weight.mean(dim=1, keepdim=True)
                  if orig_bias is not None: new_proj.bias.data = orig_bias
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
        """ Handles positional embedding interpolation. Takes BNC input. """
        if self.orig_pos_embed is None:
             _logger.warning("Original positional embedding not found. Skipping interpolation.")
             return x # No pos embed to interpolate

        # target shape for interpolation (excluding prefix tokens)
        new_grid_size = (H_grid, W_grid)
        # Ensure old_grid_size is correctly derived
        old_grid_size = (self.orig_grid_size, self.orig_grid_size)


        # Check if interpolation is needed
        if new_grid_size != old_grid_size:
            #_logger.info(f"Interpolating pos embed from {old_grid_size} to {new_grid_size}")
            pos_embed_interpolated = resample_abs_pos_embed(
                self.orig_pos_embed.to(x.device), # Ensure pos_embed is on the same device
                new_grid_size=list(new_grid_size),
                old_grid_size=list(old_grid_size),
                num_prefix_tokens=self.num_prefix_tokens,
                interpolation=self.pos_embed_interp,
                antialias=self.pos_embed_antialias,
                verbose=False, # Set to True for debugging
            )
            # Add interpolated pos embed to x (which is BNC)
            # Handle prefix tokens correctly
            x_prefix = x[:, :self.num_prefix_tokens]
            x_grid = x[:, self.num_prefix_tokens:]
            x_grid = x_grid + pos_embed_interpolated[:, self.num_prefix_tokens:, :]
            x = torch.cat([x_prefix, x_grid], dim=1)

        else:
            # Add original pos embed
            x = x + self.orig_pos_embed.to(x.device) # Ensure device match

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder Forward ---
        # 1. Patch Embedding
        x_patch_raw = self.backbone.patch_embed(x) # Output BNC: [B, N, C]
        B, N, C = x_patch_raw.shape
        # Get grid size directly after patch embedding
        H_grid, W_grid = self.backbone.patch_embed.grid_size # Use the actual grid size

        # 2. Add class token if needed by the backbone
        if hasattr(self.backbone, 'cls_token') and self.backbone.cls_token is not None:
             cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
             x_with_cls = torch.cat((cls_tokens, x_patch_raw), dim=1)
        else:
             # Make sure num_prefix_tokens is 0 if no class token
             if self.num_prefix_tokens != 0:
                  _logger.warning("Backbone has no cls_token but num_prefix_tokens > 0. Resetting to 0.")
                  self.num_prefix_tokens = 0
             x_with_cls = x_patch_raw


        # 3. Add Positional Embedding (Interpolated)
        x_with_pos = self._interpolate_pos_embed(x_with_cls, H_grid, W_grid)
        x_with_pos = self.backbone.pos_drop(x_with_pos)

        current_x_bnc = x_with_pos # Keep BNC format for blocks

        skip_connections = {} # Store skips by index

        # 4. Iterate through ViT Blocks
        for i, blk in enumerate(self.backbone.blocks):
            current_x_bnc = blk(current_x_bnc)
            # Store output if it's one of the feature indices
            if i in self.feature_indices:
                # Remove prefix tokens before reshaping for skip connection
                feature_bnc = current_x_bnc[:, self.num_prefix_tokens:, :]
                # Reshape BNC -> B, H, W, C -> B, C, H, W for decoder
                # Ensure C matches embed_dim if reshaping
                if feature_bnc.shape[-1] != C:
                     _logger.warning(f"Feature dimension {feature_bnc.shape[-1]} != Embed dim {C} at block {i}. Using embed dim C.")
                     # This case shouldn't happen with standard ViTs but added for safety
                try:
                    feature_nchw = feature_bnc.reshape(B, H_grid, W_grid, C).permute(0, 3, 1, 2).contiguous()
                    skip_connections[i] = feature_nchw
                except RuntimeError as e:
                     _logger.error(f"Error reshaping skip connection at block {i}. Shape was {feature_bnc.shape}, target grid {H_grid}x{W_grid}. Error: {e}")
                     # Handle error, maybe skip this connection or raise
                     raise e

                # _logger.debug(f"Stored skip connection from block {i}: {feature_nchw.shape}")


        # --- Decoder Forward ---
        # Ensure skip connections were extracted correctly
        if len(skip_connections) != len(self.feature_indices):
             # Try to find missing indices
             missing_indices = set(self.feature_indices) - set(skip_connections.keys())
             raise RuntimeError(f"Extracted {len(skip_connections)} skip connections ({list(skip_connections.keys())}), but expected {len(self.feature_indices)} ({list(self.feature_indices)}). Missing: {missing_indices}. Check feature_indices.")


        # Sort indices to process from deep to shallow
        sorted_indices = sorted(self.feature_indices, reverse=True)

        # Start with the deepest feature map
        decoder_x = skip_connections[sorted_indices[0]]

        # Apply decoder blocks, combining with skip connections from shallower layers
        for i in range(len(self.decoder_blocks)):
            # Skip connection comes from the next shallower feature index
            skip_index = sorted_indices[i + 1]
            skip = skip_connections[skip_index]
            decoder_x = self.decoder_blocks[i](decoder_x, skip)


        # Apply final block using the output of the last decoder stage and the shallowest skip connection
        skip_0 = skip_connections[self.feature_indices[0]] # Get shallowest skip by original index
        decoder_x = self.final_block(decoder_x, skip_0)


        # --- Final Upsampling & Output ---
        x_out = self.final_upsample(decoder_x)
        logits = self.segmentation_head(x_out)


        # Ensure final output size matches input size
        if logits.shape[2:] != x.shape[2:]:
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
        # Use create_model to get depth directly if possible
        temp_model = timm.create_model(backbone_name, features_only=False) # Create full model temporarily
        backbone_depth = len(temp_model.blocks) if hasattr(temp_model, 'blocks') else None
        del temp_model # Free memory

        if backbone_depth and max(feature_indices) >= backbone_depth:
            _logger.warning(
                f"Max feature index ({max(feature_indices)}) >= inferred backbone depth ({backbone_depth}). "
                f"Ensure indices are valid (0-based) for '{backbone_name}'."
            )
        elif not backbone_depth:
             _logger.warning(f"Could not infer depth for '{backbone_name}' via block count. Verify feature_indices.")

    except Exception as e:
        _logger.warning(f"Could not automatically determine depth for '{backbone_name}'. Please verify feature_indices. Error: {e}")

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

