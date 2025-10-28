import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import PatchEmbed
from timm.layers import resample_patch_embed, resample_abs_pos_embed_nhwc
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
        # Upsampling layer (doubles spatial resolution, halves channels)
        # Input channels = channels from below
        # Output channels = channels for this level (usually out_channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )

        # Convolution block after concatenation
        # Input channels = channels after upsampling + channels from skip connection
        # Output channels = channels for this level
        self.conv = nn.Sequential(
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
            x_below (torch.Tensor): Feature map from the layer below in the decoder.
            x_skip (torch.Tensor): Feature map from the corresponding encoder layer (skip connection).
        """
        x_up = self.upsample(x_below)

        # Ensure spatial dimensions match before concatenation
        if x_up.shape[2:] != x_skip.shape[2:]:
             # Pad or interpolate x_up if needed (common in ViT backbones)
            x_up = F.interpolate(x_up, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        x_concat = torch.cat([x_up, x_skip], dim=1)
        x_out = self.conv(x_concat)
        return x_out


class UNETR2D(nn.Module):
    """
    2D UNETR model using a timm ViT backbone.
    Extracts features from multiple ViT blocks and uses a CNN decoder.
    """
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224',
        img_size: int = 1024,
        in_chans: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
        # Indices of ViT blocks to extract features from (adjust based on backbone depth)
        feature_indices: Tuple[int, ...] = (2, 5, 8, 11),
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32), # Channels for decoder stages
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_indices = feature_indices
        self.img_size = img_size
        self.in_chans = in_chans

        # --- Encoder (ViT Backbone) ---
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            img_size=img_size, # Pass img_size for pos embed resizing
            # Request features from specific blocks AND the final output
            features_only=False, # We need intermediate blocks AND final output for projection
            # out_indices=feature_indices # timm ViT features_only doesn't work well with intermediate indices
        )

        # Manually adjust patch embedding if input channels don't match pretrained
        if in_chans != self.backbone.patch_embed.proj.in_channels:
             _logger.warning(
                 f"Input channels ({in_chans}) mismatch pretrained ({self.backbone.patch_embed.proj.in_channels}). "
                 f"Adapting patch embedding projection."
             )
             orig_weight = self.backbone.patch_embed.proj.weight
             orig_bias = self.backbone.patch_embed.proj.bias
             new_proj = nn.Conv2d(
                 in_chans,
                 self.backbone.embed_dim,
                 kernel_size=self.backbone.patch_embed.patch_size,
                 stride=self.backbone.patch_embed.stride,
                 padding=self.backbone.patch_embed.padding
             )

             # Adapt weights (simple averaging if going from 3 to 1)
             if in_chans == 1 and orig_weight.shape[1] == 3:
                 new_proj.weight.data = orig_weight.data.mean(dim=1, keepdim=True)
             else:
                 _logger.warning("Could not automatically adapt patch embedding weights for channels. Using random init.")
                 # Keep random init or implement more sophisticated adaptation

             if orig_bias is not None:
                 new_proj.bias.data = orig_bias.data
             self.backbone.patch_embed.proj = new_proj

        # Check patch size compatibility and potentially resize patch embed kernel
        current_patch_size = self.backbone.patch_embed.patch_size[0]
        encoder_embed_dim = self.backbone.embed_dim

        # --- Get feature info (channel dims) ---
        # We run a dummy forward pass to get intermediate shapes if needed,
        # or infer from known architectures. For standard ViT: all stages have embed_dim.
        # UNETR typically projects features before the decoder.
        # Let's assume features from blocks have embed_dim.
        skip_channel_dims = [encoder_embed_dim] * len(feature_indices)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Decoder input starts from the deepest selected feature
        decoder_in_channels = skip_channel_dims[-1]

        # Build decoder blocks from deep to shallow
        for i in range(len(feature_indices) - 1, 0, -1):
            skip_ch = skip_channel_dims[i-1]
            out_ch = decoder_channels[i] # Output channels for this decoder stage
            self.decoder_blocks.append(
                UNETRDecoderBlock(decoder_in_channels, skip_ch, out_ch)
            )
            decoder_in_channels = out_ch # Input for the next block is the output of this one

        # --- Final Upsampling and Output ---
        # The shallowest skip connection (usually from early blocks or patch embed)
        # Needs to be upsampled to the original image size.
        # The input to the final upsampling is the output of the last decoder block.
        final_up_in_channels = decoder_in_channels
        # Output channels before the final segmentation layer
        final_up_out_channels = decoder_channels[0]

        # Upsample to match the shallowest feature map (often patch embed output)
        # Then another block to combine with the shallowest skip
        self.final_block = UNETRDecoderBlock(
            final_up_in_channels, skip_channel_dims[0], final_up_out_channels
        )

        # Additional upsampling needed to reach original image size
        # Calculate the required upsampling factor from the shallowest feature map
        # ViT patch embedding output is H/P x W/P
        final_upsample_factor = self.backbone.patch_embed.patch_size[0] // 2 # Since final_block already upsamples by 2

        upsample_layers = []
        current_channels = final_up_out_channels
        num_final_upsamples = int(torch.log2(torch.tensor(float(final_upsample_factor))).item())

        for _ in range(num_final_upsamples):
             # You might want to decrease channels here too
            upsample_layers.append(nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=2, stride=2))
            upsample_layers.append(nn.BatchNorm2d(current_channels // 2))
            upsample_layers.append(nn.ReLU(inplace=True))
            current_channels //= 2

        self.final_upsample = nn.Sequential(*upsample_layers)

        # Final 1x1 convolution
        self.segmentation_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder Forward ---
        # Need to modify ViT forward to get intermediate block outputs
        # Standard timm ViT `forward_features` goes through all blocks.
        # We'll manually iterate through blocks.

        x_patch = self.backbone.patch_embed(x)
        B, H_grid, W_grid, C = x_patch.shape

        # Add positional embedding
        if self.backbone.pos_embed is not None:
             # Dynamically resize abs pos embedding
            pos_embed_resized = resample_abs_pos_embed_nhwc(self.backbone.pos_embed, (H_grid, W_grid))
            x_patch = x_patch + pos_embed_resized

        x_patch = self.backbone.pos_drop(x_patch)
        x_patch = self.backbone.norm_pre(x_patch)

        skip_connections = []
        current_x = x_patch
        block_counter = 0

        # Iterate through blocks and store outputs at specified indices
        # Assuming self.backbone.blocks is iterable (nn.Sequential or nn.ModuleList)
        for i, blk in enumerate(self.backbone.blocks):
            current_x = blk(current_x)
            if i in self.feature_indices:
                # Features need to be in NCHW format for CNN decoder
                skip_connections.append(current_x.permute(0, 3, 1, 2).contiguous())

        # --- Decoder Forward ---
        # Start with the deepest feature map
        decoder_x = skip_connections[-1]

        # Apply decoder blocks, combining with skip connections from shallower layers
        # Iterate skip connections from second deepest to shallowest
        for i in range(len(self.decoder_blocks)):
            skip = skip_connections[len(skip_connections) - 2 - i]
            decoder_x = self.decoder_blocks[i](decoder_x, skip)

        # Apply final block with the shallowest skip connection
        # decoder_x = self.final_block(decoder_x, skip_connections[0]) # Assuming this logic

        # Alternative: Maybe the final block just processes the output of the last decoder_block
        # Let's follow a simpler path first:
        # Pass the output of the last standard decoder block to final upsampling

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
    logging.basicConfig(level=logging.INFO)

    img_size = 1024
    in_chans = 1
    num_classes = 1

    # Using ViT-Base with patch size 16 (adjust if needed, e.g., vit_small_patch16_224)
    model = create_unetr_model(
        backbone_name='vit_base_patch16_224', # Standard ViT-B backbone
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=True, # Load pretrained weights for the backbone
        # Feature indices might need adjustment if using a different depth ViT
        feature_indices=(2, 5, 8, 11), # Indices for ViT-Base (depth 12)
        decoder_channels=(256, 128, 64, 32), # Example channel configuration
    )

    print(f"UNETR Model created successfully.")

    # Test forward pass
    dummy_input = torch.randn(1, in_chans, img_size, img_size)
    print(f"Input shape: {dummy_input.shape}")

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        assert output.shape == (1, num_classes, img_size, img_size), "Output shape mismatch!"
        print("Forward pass successful and output shape is correct.")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        # Print model structure for debugging if forward pass fails
        # print("\nModel Structure:")
        # print(model)
