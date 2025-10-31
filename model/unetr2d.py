import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import PatchEmbed
from timm.layers import resample_patch_embed, PatchEmbed, Mlp, DropPath, lecun_normal_, trunc_normal_
import logging
from typing import Tuple, Optional, List
from torch.hub import load_state_dict_from_url # 导入用于加载权重的函数

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
    Dynamically creates patch_embed based on target_patch_grid_size.
    """
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224',
        img_size: int = 1024,
        in_chans: int = 1,
        num_classes: int = 1,
        pretrained: bool = False, # Pretrained weights are loaded in the factory fn
        feature_indices: Tuple[int, ...] = (2, 5, 8, 11),
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
        target_patch_grid_size: int = 64, # 目标网格大小
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

        # --- **修改点: 动态计算 Patch Size** ---
        self.patch_size = img_size // target_patch_grid_size
        if img_size % target_patch_grid_size != 0:
            raise ValueError(f"Image size {img_size} is not divisible by target grid size {target_patch_grid_size}")
        _logger.info(f"UNETR: img_size={img_size}, grid_size={target_patch_grid_size} -> Calculated patch_size={self.patch_size}")

        # --- Encoder (ViT Backbone) ---
        # 1. 创建骨干网络架构 (不加载权重)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False, # 权重将在工厂函数中手动加载和调整
            in_chans=in_chans,
            img_size=img_size,
            features_only=False,
        )
        
        # 2. 获取原始 (预训练) 模型的 pos_embed 和 grid_size
        # (基于 vit_base_patch16_224)
        self.num_prefix_tokens = self.backbone.num_prefix_tokens # e.g., 1 for CLS token
        self.orig_grid_size = 14 # vit_base_patch16_224 (224 / 16 = 14)
        orig_pos_embed_size = (self.orig_grid_size ** 2) + self.num_prefix_tokens
        # 创建一个 nn.Parameter 来存储 *原始的* pos_embed，以便加载
        self.orig_pos_embed = nn.Parameter(
            torch.zeros(1, orig_pos_embed_size, self.backbone.embed_dim)
        )
        self.orig_pos_embed.data.normal_(mean=0.0, std=0.02) # Standard init
        trunc_normal_(self.orig_pos_embed, std=.02) # Re-init like timm

        # 3. **替换** backbone 的 patch_embed 为我们自定义的
        encoder_embed_dim = self.backbone.embed_dim
        self.backbone.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size, # 使用我们计算出的 patch size
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
        )

        # 4. **替换** backbone 的 pos_embed 为 None
        # 我们将在 forward 传递中手动插值和添加 self.orig_pos_embed
        self.backbone.pos_embed = None 
        
        # 5. 获取新的 grid_size
        self.grid_size = self.backbone.patch_embed.grid_size
        _logger.info(f"New patch_embed created with grid_size {self.grid_size}")


        # --- Skip Connection Feature Dimensions ---
        skip_channel_dims = [encoder_embed_dim] * len(feature_indices)

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = skip_channel_dims[-1]
        if len(decoder_channels) != len(feature_indices):
            raise ValueError(f"decoder_channels 长度 ({len(decoder_channels)}) 必须匹配 feature_indices ({len(feature_indices)})")

        for i in range(len(feature_indices) - 1, 0, -1):
            skip_ch = skip_channel_dims[i-1]
            out_ch = decoder_channels[i]
            self.decoder_blocks.append(
                UNETRDecoderBlock(decoder_in_channels, skip_ch, out_ch)
            )
            decoder_in_channels = out_ch

        # --- Final Blocks ---
        self.final_block = UNETRDecoderBlock(
            decoder_in_channels, skip_channel_dims[0], decoder_channels[0]
        )

        # 最终上采样以匹配原始图像尺寸
        final_up_factor = self.patch_size // 2 # final_block 已经上采样了 2x
        num_final_upsamples = int(torch.log2(torch.tensor(float(final_up_factor))).item()) if final_up_factor > 1 else 0

        upsample_layers = []
        current_channels = decoder_channels[0]
        for _ in range(num_final_upsamples):
            next_channels = max(16, current_channels // 2)
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

        new_grid_size = (H_grid, W_grid)
        old_grid_size = (self.orig_grid_size, self.orig_grid_size)

        if new_grid_size != old_grid_size:
            pos_embed_interpolated = resample_abs_pos_embed(
                self.orig_pos_embed.to(x.device),
                new_grid_size=list(new_grid_size),
                old_grid_size=list(old_grid_size),
                num_prefix_tokens=self.num_prefix_tokens,
                interpolation=self.pos_embed_interp,
                antialias=self.pos_embed_antialias,
                verbose=False,
            )
            x = x + pos_embed_interpolated
        else:
            x = x + self.orig_pos_embed.to(x.device)

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Encoder Forward ---
        # 1. Patch Embedding
        x_patch_raw = self.backbone.patch_embed(x) # Output BNC: [B, N, C]
        B, N, C = x_patch_raw.shape
        H_grid, W_grid = self.backbone.patch_embed.grid_size # Get grid size

        # 2. Add class token if needed
        if hasattr(self.backbone, 'cls_token') and self.backbone.cls_token is not None:
             cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
             x_with_cls = torch.cat((cls_tokens, x_patch_raw), dim=1)
        else:
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
            if i in self.feature_indices:
                feature_bnc = current_x_bnc[:, self.num_prefix_tokens:, :]
                try:
                    feature_nchw = feature_bnc.reshape(B, H_grid, W_grid, C).permute(0, 3, 1, 2).contiguous()
                    skip_connections[i] = feature_nchw
                except RuntimeError as e:
                     _logger.error(f"Error reshaping skip connection at block {i}. Shape was {feature_bnc.shape}, target grid {H_grid}x{W_grid}. Error: {e}")
                     raise e
        
        # --- Decoder Forward ---
        if len(skip_connections) != len(self.feature_indices):
             missing_indices = set(self.feature_indices) - set(skip_connections.keys())
             raise RuntimeError(f"Extracted {len(skip_connections)} skip connections, expected {len(self.feature_indices)}. Missing: {missing_indices}.")

        sorted_indices = sorted(self.feature_indices, reverse=True)
        decoder_x = skip_connections[sorted_indices[0]]

        for i in range(len(self.decoder_blocks)):
            skip_index = sorted_indices[i + 1]
            skip = skip_connections[skip_index]
            decoder_x = self.decoder_blocks[i](decoder_x, skip)

        skip_0 = skip_connections[self.feature_indices[0]] # Get shallowest skip
        decoder_x = self.final_block(decoder_x, skip_0)

        # --- Final Upsampling & Output ---
        x_out = self.final_upsample(decoder_x)
        logits = self.segmentation_head(x_out)

        # Ensure final output size matches input size
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits

# --- *** 新的工厂函数，用于加载和调整权重 *** ---
def create_unetr_model(
    backbone_name: str = 'vit_base_patch16_224',
    img_size: int = 1024,
    in_chans: int = 1,
    num_classes: int = 1,
    pretrained: bool = True,
    target_patch_grid_size: int = 64, # 目标网格大小
    feature_indices: Tuple[int, ...] = (2, 5, 8, 11),
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
) -> UNETR2D:
    """
    Creates a UNETR-2D model with a specified timm ViT backbone,
    with dynamic patch size adjustment and weight interpolation.
    """
    
    # 1. 创建模型架构 (pretrained=False)
    model = UNETR2D(
        backbone_name=backbone_name,
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=False, # We load weights manually
        feature_indices=feature_indices,
        decoder_channels=decoder_channels,
        target_patch_grid_size=target_patch_grid_size,
    )

    # 2. 如果需要，加载和调整预训练权重
    if pretrained:
        _logger.info(f"Loading and adapting pretrained weights for {backbone_name}")
        # 获取预训练配置
        default_cfg = timm.get_pretrained_cfg(backbone_name)
        if not default_cfg or not default_cfg.url:
            _logger.warning(f"No pretrained URL found for {backbone_name}. Skipping weight loading.")
            return model
            
        # 从 URL 加载权重
        state_dict = load_state_dict_from_url(default_cfg.url, map_location='cpu')

        # --- 调整 Patch Embedding ---
        orig_patch_embed_key = 'patch_embed.proj.weight'
        if orig_patch_embed_key in state_dict:
            orig_weight = state_dict[orig_patch_embed_key]
            new_patch_size = model.patch_size # 从模型实例中获取计算出的 patch size
            orig_patch_size = 16 # 假设 'vit_base_patch16_224'
            
            target_weight = orig_weight
            
            # 1. 插值 Patch Size
            if new_patch_size != orig_patch_size:
                _logger.info(f"Interpolating patch_embed from {orig_patch_size} to {new_patch_size}")
                target_weight = F.interpolate(
                    target_weight, size=(new_patch_size, new_patch_size),
                    mode='bicubic', align_corners=False
                )

            # 2. 调整通道
            orig_in_chans = target_weight.shape[1]
            if in_chans != orig_in_chans:
                _logger.info(f"Adapting patch_embed from {orig_in_chans} to {in_chans} channels")
                if in_chans == 1 and orig_in_chans == 3:
                    target_weight = target_weight.mean(dim=1, keepdim=True)
                elif in_chans == 3 and orig_in_chans == 1:
                    target_weight = target_weight.repeat(1, 3, 1, 1)
                else:
                    _logger.warning(f"Unsupported channel conversion: {orig_in_chans} -> {in_chans}. Using random init for patch_embed.")
                    del state_dict[orig_patch_embed_key] # Remove key to use random init

            if orig_patch_embed_key in state_dict:
                state_dict[orig_patch_embed_key] = target_weight

            # 3. 处理 Bias (如果存在)
            orig_bias_key = 'patch_embed.proj.bias'
            if orig_bias_key in state_dict:
                # 权重插值后，bias 通常保持不变
                pass
                
        # --- 加载原始 Positional Embedding ---
        # 我们将 state_dict 中的 'pos_embed' 加载到 model.orig_pos_embed
        if 'pos_embed' in state_dict:
            # 确保尺寸匹配
            if state_dict['pos_embed'].shape == model.orig_pos_embed.shape:
                 model.orig_pos_embed.data.copy_(state_dict['pos_embed'])
                 # 从 state_dict 中移除，因为它不匹配 model.backbone.pos_embed (后者是 None)
                 del state_dict['pos_embed']
            else:
                 _logger.error(f"Checkpoint pos_embed shape {state_dict['pos_embed'].shape} "
                               f"mismatches model.orig_pos_embed shape {model.orig_pos_embed.shape}. Skipping pos_embed loading.")
                 del state_dict['pos_embed']
        else:
            _logger.warning("No 'pos_embed' found in checkpoint.")

        # --- 加载所有其他权重 (Blocks, Norms, etc.) ---
        # 我们加载到 model.backbone 中
        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        _logger.info(f"Backbone weights loaded. Missing keys: {missing}. Unexpected keys: {unexpected}")

    return model

# --- Local Test ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Use INFO for cleaner output

    # --- 1k 测试 ---
    _logger.info("\n--- 1. 测试 1k 图像 (1024x1024) ---")
    img_size_1k = 1024
    in_chans_1k = 1
    grid_size_1k = 64 # 1024 / 64 = 16 (patch_size)
    model_1k = create_unetr_model(
        backbone_name='vit_base_patch16_224',
        img_size=img_size_1k,
        in_chans=in_chans_1k,
        num_classes=5,
        pretrained=True,
        target_patch_grid_size=grid_size_1k,
        feature_indices=(2, 5, 8, 11),
        decoder_channels=(256, 128, 64, 32),
    )
    _logger.info(f"UNETR 1k Model (patch_size={model_1k.patch_size}) created.")
    dummy_input_1k = torch.randn(1, in_chans_1k, img_size_1k, img_size_1k)
    try:
        model_1k.eval()
        with torch.no_grad(): output = model_1k(dummy_input_1k)
        _logger.info(f"1k Input: {dummy_input_1k.shape} -> Output: {output.shape}")
        assert output.shape == (1, 5, img_size_1k, img_size_1k)
        _logger.info("1k 测试通过。")
    except Exception as e: _logger.error(f"1k 测试失败: {e}", exc_info=True)


    # --- 8k 测试 ---
    _logger.info("\n--- 2. 测试 8k 图像 (8192x8192) ---")
    img_size_8k = 8192
    in_chans_8k = 1
    grid_size_8k = 64 # 8192 / 64 = 128 (patch_size)
    model_8k = create_unetr_model(
        backbone_name='vit_base_patch16_224',
        img_size=img_size_8k,
        in_chans=in_chans_8k,
        num_classes=5,
        pretrained=True,
        target_patch_grid_size=grid_size_8k,
        feature_indices=(2, 5, 8, 11),
        decoder_channels=(256, 128, 64, 32),
    )
    _logger.info(f"UNETR 8k Model (patch_size={model_8k.patch_size}) created.")
    dummy_input_8k = torch.randn(1, in_chans_8k, img_size_8k, img_size_8k)
    try:
        model_8k.eval()
        with torch.no_grad(): output = model_8k(dummy_input_8k)
        _logger.info(f"8k Input: {dummy_input_8k.shape} -> Output: {output.shape}")
        assert output.shape == (1, 5, img_size_8k, img_size_8k)
        _logger.info("8k 测试通过。")
    except Exception as e: _logger.error(f"8k 测试失败: {e}", exc_info=True)

