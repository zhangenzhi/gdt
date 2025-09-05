import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block

# 注意：我们不再需要从 mae_timm 导入 MAEEncoder
# from model.mae_timm import MAEEncoder 

class PromptEncoder(nn.Module):
    """
    一个简化的提示编码器，处理稀疏的点提示。(此部分保持不变)
    """
    def __init__(self, embed_dim, num_point_embeddings=4):
        super().__init__()
        self.point_embeddings = nn.Embedding(num_point_embeddings, embed_dim)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def forward(self, points, labels):
        """
        points: (B, num_points, 2) 坐标
        labels: (B, num_points) 标签
        """
        B, N, _ = points.shape
        point_embed = self.point_embeddings(labels)
        if N == 0:
            return self.not_a_point_embed.weight.unsqueeze(0).expand(B, -1, -1)
        return point_embed

class MaskDecoder(nn.Module):
    """
    一个简化的掩码解码器。(此部分保持不变)
    """
    def __init__(self, encoder_dim, decoder_dim, num_classes, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Linear(decoder_dim, decoder_dim // 4),
            nn.Linear(decoder_dim // 4, decoder_dim)
        ])
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(encoder_dim, encoder_dim // 4, kernel_size=2, stride=2),
            # 注意：这里的LayerNorm尺寸可能需要根据patch_size调整，这是一个简化实现
            nn.LayerNorm([encoder_dim // 4, patch_size // 2, patch_size // 2]),
            nn.GELU(),
            nn.ConvTranspose2d(encoder_dim // 4, encoder_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(encoder_dim // 8, num_classes, kernel_size=1)
        )
        self.blocks = nn.ModuleList([
            Block(dim=encoder_dim, num_heads=8, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(2)
        ])

    def forward(self, image_embeddings, prompt_embeddings):
        # 融合特征
        # 简化融合：直接将点的embedding广播并添加到图像特征上
        prompt_embed = prompt_embeddings[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        fused_features = image_embeddings + prompt_embed
        
        # 通过 Transformer blocks
        for blk in self.blocks:
            fused_features = blk(fused_features)
            
        upscaled_masks = self.output_upscaling(fused_features)
        return upscaled_masks

class SAMLikeModel(nn.Module):
    """
    使用 timm 的 VisionTransformerSam 作为骨干网络的 SAM-like 模型
    """
    def __init__(self, config, num_classes):
        super().__init__()
        self.patch_size = config['model']['patch_size']
        
        # --- 关键修改: 直接使用 timm 的 SAM ViT 模型 ---
        # 1. 图像编码器 (使用您预训练的 ViT)
        self.image_encoder = timm.create_model(
            'vit_base_patch16_sam',  # timm 中官方的 SAM ViT-Base 模型
            pretrained=False,        # 我们将加载自己的预训练权重
            img_size=config['model']['img_size'],
            in_chans=config['model']['in_channels']
        )
        
        # 从创建的 timm 模型中动态获取 embedding 维度
        encoder_dim = self.image_encoder.embed_dim
        
        # 2. 提示编码器 (使用新的 encoder_dim)
        self.prompt_encoder = PromptEncoder(embed_dim=encoder_dim)
        
        # 3. 掩码解码器 (使用新的 encoder_dim)
        self.mask_decoder = MaskDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=config['model']['decoder_embed_dim'],
            num_classes=num_classes,
            patch_size=self.patch_size
        )

    def forward(self, images, points, labels):
        H, W = images.shape[2], images.shape[3]
        
        # --- 关键修改: 使用 image_encoder 的 forward_features ---
        # 1. 提取图像特征, timm 的 SAM 模型不使用 [CLS] token
        image_features_seq = self.image_encoder.forward_features(images)
        
        # 将序列特征恢复为 2D 图像特征
        B, N, D = image_features_seq.shape
        h_p, w_p = H // self.patch_size, W // self.patch_size
        image_embeddings_2d = image_features_seq.permute(0, 2, 1).reshape(B, D, h_p, w_p)
        
        # 2. 编码提示
        prompt_embeddings = self.prompt_encoder(points, labels)
        
        # 3. 解码掩码
        predicted_masks = self.mask_decoder(image_embeddings_2d, prompt_embeddings)
        
        # 将最终输出上采样到原始图像大小
        predicted_masks = F.interpolate(
            predicted_masks,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return predicted_masks

    def load_vit_backbone(self, mae_checkpoint_path):
        """
        从预训练的 MAE 模型中加载 Image Encoder 的权重。
        此函数会处理权重键名的前缀，以匹配新的 timm 模型。
        """
        checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
        mae_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # --- 关键修改: 调整权重键名以匹配 timm 模型 ---
        encoder_state_dict = {}
        # 预训练权重的键名类似: 'encoder.model.blocks.0...'
        # timm 模型的键名类似: 'blocks.0...'
        prefix_to_strip = 'encoder.model.'
        
        for k, v in mae_state_dict.items():
            if k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
                encoder_state_dict[new_key] = v
        
        # 加载权重到 image_encoder
        msg = self.image_encoder.load_state_dict(encoder_state_dict, strict=False)
        print("加载 ViT backbone 的消息:", msg)

# ================================================================= #
#                   单文件测试模块 (Single File Test)                  #
# ================================================================= #
if __name__ == '__main__':
    print("--- [测试] SAM-like 模型单文件 ---")

    # 1. 模拟您的配置文件 (config)
    mock_config = {
        'model': {
            'img_size': 8192,
            'patch_size': 128,    # 根据您的 MAE 预训练设置
            'in_channels': 1,     # 单通道灰度图
            'encoder_embed_dim': 768,
            'decoder_embed_dim': 512,
            # 其他 encoder/decoder 参数, 模型初始化时会用到
            'encoder_depth': 12,
            'encoder_heads': 12,
            'decoder_depth': 8,
            'decoder_heads': 16,
        }
    }
    
    # 2. 设置模型参数
    NUM_CLASSES = 5 # 您指定的类别数
    BATCH_SIZE = 1  # 测试时通常用1
    IMG_SIZE = mock_config['model']['img_size']
    
    # 3. 实例化模型
    # 将模型移动到GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMLikeModel(config=mock_config, num_classes=NUM_CLASSES).to(device)
    model.eval() # 设置为评估模式
    
    print(f"模型已成功实例化并移至: {device}")
    
    # 4. 创建模拟输入数据
    # a. 模拟图像: [B, C, H, W]
    mock_image = torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(device)
    
    # b. 模拟提示: 一个前景点 (坐标 + 标签)
    #    - 坐标: [B, Num_points, 2] (x, y)
    #    - 标签: [B, Num_points] (1 通常代表前景点)
    mock_points = torch.randint(0, IMG_SIZE, (BATCH_SIZE, 1, 2), device=device).float()
    mock_labels = torch.ones(BATCH_SIZE, 1, device=device).long()

    print("\n--- 输入数据尺寸 ---")
    print(f"图像 (Image):  {mock_image.shape}")
    print(f"点坐标 (Points): {mock_points.shape}")
    print(f"点标签 (Labels): {mock_labels.shape}")

    # 5. 执行前向传播
    with torch.no_grad(): # 测试时不需要计算梯度
        try:
            predicted_masks = model(mock_image, mock_points, mock_labels)
            print("\n--- 前向传播成功 ---")
            print(f"输出掩码 (Predicted Masks) 尺寸: {predicted_masks.shape}")
            
            # 验证输出尺寸是否符合预期
            expected_shape = (BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
            assert predicted_masks.shape == expected_shape, \
                f"输出尺寸错误！预期: {expected_shape}, 得到: {predicted_masks.shape}"
            
            print("\n✅ 测试通过: 输出尺寸符合预期！")

        except Exception as e:
            print(f"\n❌ 测试失败: 前向传播时发生错误。")
            print(f"错误信息: {e}")

