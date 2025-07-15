import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Tuple
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# 假设 HierarchicalViTEncoder 在 gdt_vit.py 中定义
from gdt.gdt import HierarchicalViTEncoder, SinusoidalPositionalEncoder
from model.vit import  RelativeTransformerBlock

# ======================================================================
# MAE 解码器和顶层模型
# ======================================================================

class MAEDecoder(nn.Module):
    """一个轻量级的 Transformer 解码器，用于从可见 token 重建被遮盖的 token。"""
    def __init__(self, *, encoder_embed_dim: int, decoder_embed_dim: int, depth: int, num_heads: int, num_patches: int, patch_size: int, in_channels: int, mlp_ratio=4.0):
        super().__init__()
        self.decoder_embed_dim = decoder_embed_dim
        
        # 将编码器输出的 token 投影到解码器的维度
        self.embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        # 解码器的可学习 [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 解码器的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=num_heads, dim_feedforward=int(decoder_embed_dim*mlp_ratio), batch_first=True)
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        
        # 将解码后的 token 投影回像素空间
        self.head = nn.Linear(decoder_embed_dim, in_channels * patch_size * patch_size)
    
    def forward(self, visible_tokens: torch.Tensor, visible_indices: torch.Tensor):
        B, N_visible, D_encoder = visible_tokens.shape
        N_total = self.pos_embed.shape[1]
        
        # 1. 投影并准备解码器的输入序列
        x = self.embed(visible_tokens)
        
        decoder_input = self.mask_token.expand(B, N_total, -1)
        index = visible_indices.unsqueeze(-1).expand(B, N_visible, self.decoder_embed_dim)
        decoder_input = decoder_input.scatter(1, index, x)
        
        # 2. 添加位置编码
        decoder_input = decoder_input + self.pos_embed
        
        # 3. 通过解码器
        decoded_tokens = self.decoder_transformer(decoder_input)
        decoded_tokens = self.norm(decoded_tokens)
        
        # 4. 投影回头像素空间
        recon_patches = self.head(decoded_tokens)
        return recon_patches

class HierarchicalMAE(nn.Module):
    """顶层模型，封装了层级式编码器和 MAE 解码器，专用于预训练。"""
    def __init__(self, encoder: HierarchicalViTEncoder, decoder: MAEDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, img: torch.Tensor):
        # 1. 通过编码器获取可见 token 和叶子结点信息
        encoder_output = self.encoder(img)
        visible_tokens = encoder_output['visible_tokens_for_decoder']
        visible_indices = encoder_output['visible_indices']
        masked_indices = encoder_output['masked_indices']
        target_patches = encoder_output['target_patches']
        
        # 2. 通过解码器重建所有 patch
        reconstructed_patches_all = self.decoder(visible_tokens, visible_indices)
        
        # 3. 只在被遮盖的 patch 上计算损失
        B, N_total, D_pixel = reconstructed_patches_all.shape
        
        recon_masked = torch.gather(reconstructed_patches_all, 1, masked_indices.unsqueeze(-1).expand(-1, -1, D_pixel))
        target_masked = torch.gather(target_patches, 1, masked_indices.unsqueeze(-1).expand(-1, -1, D_pixel))
        
        loss = F.mse_loss(recon_masked, target_masked)
        
        return loss, reconstructed_patches_all, visible_indices

# ======================================================================
# 模型创建和可视化函数
# ======================================================================

def create_gdt_mae(config: Dict) -> HierarchicalMAE:
    """
    一个用于创建 HierarchicalMAE 模型的工厂函数。
    """
    encoder_cfg = config['encoder']
    decoder_cfg = config['decoder']
    initial_patch_size = encoder_cfg['stages'][0]['patch_size_in']
    num_initial_patches = (encoder_cfg['img_size'] // initial_patch_size)**2

    encoder = HierarchicalViTEncoder(
        img_size=encoder_cfg['img_size'], 
        stages_config=encoder_cfg['stages'], 
        embed_dim=encoder_cfg['embed_dim'], 
        num_heads=encoder_cfg['num_heads'], 
        in_channels=encoder_cfg['in_channels']
    )
    
    decoder = MAEDecoder(
        encoder_embed_dim=encoder_cfg['embed_dim'],
        decoder_embed_dim=decoder_cfg['embed_dim'],
        depth=decoder_cfg['depth'],
        num_heads=decoder_cfg['num_heads'],
        num_patches=num_initial_patches,
        patch_size=initial_patch_size,
        in_channels=encoder_cfg['in_channels']
    )
    
    model = HierarchicalMAE(encoder, decoder)
    return model

def visualize_reconstruction(img_tensor: torch.Tensor, recon_patches: torch.Tensor, visible_indices: torch.Tensor, initial_patch_size: int, output_filename: str):
    """生成并保存重建的可视化图像。"""
    print("开始生成重建可视化图像...")
    
    # 将输入 tensor 转换为 PIL Image
    # 假设输入 tensor 是标准化的，我们需要逆标准化
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )
    original_img_tensor = inv_normalize(img_tensor[0].cpu())
    original_img = transforms.ToPILImage()(original_img_tensor.clamp(0, 1))
    img_size = original_img.size
    
    # 1. 重建图片
    recon_patches_img_format = recon_patches.transpose(1, 2)
    folder = nn.Fold(output_size=img_size, kernel_size=initial_patch_size, stride=initial_patch_size)
    recon_img_tensor = folder(recon_patches_img_format)
    recon_img_tensor_normalized = (recon_img_tensor - recon_img_tensor.min()) / (recon_img_tensor.max() - recon_img_tensor.min() + 1e-6)
    recon_img = transforms.ToPILImage()(recon_img_tensor_normalized[0].cpu())
    
    # 2. 创建 Masked 图片
    masked_img = Image.new('RGB', img_size, (0,0,0))
    grid_size = img_size[0] // initial_patch_size
    visible_indices_set = set(visible_indices[0].cpu().numpy())
    
    for i in range(grid_size * grid_size):
        if i in visible_indices_set:
            row, col = i // grid_size, i % grid_size
            box = (col * initial_patch_size, row * initial_patch_size, (col+1)*initial_patch_size, (row+1)*initial_patch_size)
            patch = original_img.crop(box)
            masked_img.paste(patch, box)

    # 3. 拼接图片
    w, h = img_size
    total_width = w * 3 + 20
    result_img = Image.new('RGB', (total_width, h), (255, 255, 255))
    result_img.paste(original_img, (0, 0))
    result_img.paste(masked_img, (w + 10, 0))
    result_img.paste(recon_img, (w * 2 + 20, 0))
    
    draw = ImageDraw.Draw(result_img)
    try: font = ImageFont.truetype("arial.ttf", 14)
    except IOError: font = ImageFont.load_default()
    draw.text((10, 10), "Original", fill="black", font=font)
    draw.text((w + 20, 10), "Masked (Visible to Encoder)", fill="black", font=font)
    draw.text((w * 2 + 30, 10), "Reconstructed by Decoder", fill="black", font=font)

    result_img.save(output_filename)
    print(f"重建可视化结果已保存到文件: {output_filename}")
