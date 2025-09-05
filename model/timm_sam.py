import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block

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
    一个简化的掩码解码器。
    """
    def __init__(self, encoder_dim, decoder_dim, num_classes, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        
        # --- 核心修正 ---
        # 动态计算 LayerNorm 所需的 H 和 W 维度
        # 1. 计算 Encoder 输出的特征图网格大小
        grid_size = img_size // patch_size
        # 2. 第一次 ConvTranspose2d (stride=2) 后，网格大小翻倍
        upsampled_grid_size = grid_size * 2
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(encoder_dim, encoder_dim // 4, kernel_size=2, stride=2),
            # 使用动态计算出的正确尺寸
            nn.LayerNorm([encoder_dim // 4, upsampled_grid_size, upsampled_grid_size]),
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
        prompt_embed = prompt_embeddings[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        fused_features = image_embeddings + prompt_embed
        
        B, C, H, W = fused_features.shape
        fused_features_seq = fused_features.flatten(2).transpose(1, 2) # Shape: [B, H*W, C]
        
        for blk in self.blocks:
            fused_features_seq = blk(fused_features_seq)
            
        processed_features = fused_features_seq.transpose(1, 2).view(B, C, H, W)
        
        upscaled_masks = self.output_upscaling(processed_features)
        return upscaled_masks

class SAMLikeModel(nn.Module):
    """
    使用 timm 的 VisionTransformerSam 作为骨干网络的 SAM-like 模型 (最终版)
    """
    def __init__(self, config, num_classes):
        super().__init__()
        model_config = config['model']
        self.patch_size = model_config['patch_size']
        self.img_size = model_config['img_size']
        
        self.image_encoder = timm.create_model(
            'samvit_base_patch16.sa1b',
            pretrained=False,
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=model_config['in_channels'],
            embed_dim=model_config['encoder_embed_dim'],
            depth=model_config['encoder_depth'],
            num_heads=model_config['encoder_heads'],
        )
        
        encoder_output_dim = self.image_encoder.num_features
        
        self.prompt_encoder = PromptEncoder(embed_dim=encoder_output_dim)
        
        self.mask_decoder = MaskDecoder(
            encoder_dim=encoder_output_dim,
            decoder_dim=model_config['decoder_embed_dim'],
            num_classes=num_classes,
            patch_size=self.patch_size,
            img_size=self.img_size # <-- 将 img_size 传递给解码器
        )

    def forward(self, images, points, labels):
        H, W = images.shape[2], images.shape[3]
        
        image_embeddings_2d = self.image_encoder.forward_features(images)
        
        prompt_embeddings = self.prompt_encoder(points, labels)
        predicted_masks = self.mask_decoder(image_embeddings_2d, prompt_embeddings)
        
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
        """
        checkpoint = torch.load(mae_checkpoint_path, map_location='cpu')
        mae_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        encoder_state_dict = {}
        prefix_to_strip = 'encoder.model.'
        
        for k, v in mae_state_dict.items():
            if k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
                encoder_state_dict[new_key] = v
        
        msg = self.image_encoder.load_state_dict(encoder_state_dict, strict=False)
        print("加载 ViT backbone 的消息:", msg)

# ================================================================= #
#          单文件测试模块 (Single File Test - Forward & Backward)       #
# ================================================================= #
if __name__ == '__main__':
    print("--- [测试] SAM-like 模型单文件 (前向 + 后向传播) ---")

    # 1. 模拟您的配置文件 (config)
    mock_config = {
        'model': {
            'img_size': 8192,
            'patch_size': 128,
            'in_channels': 1,
            'encoder_embed_dim': 768,
            'decoder_embed_dim': 512,
            'encoder_depth': 12,
            'encoder_heads': 12,
            'decoder_depth': 8,
            'decoder_heads': 16,
        }
    }
    
    # 2. 设置模型参数
    NUM_CLASSES = 5
    BATCH_SIZE = 1
    IMG_SIZE = mock_config['model']['img_size']
    
    # 3. 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAMLikeModel(config=mock_config, num_classes=NUM_CLASSES).to(device)
    
    print(f"模型已成功实例化并移至: {device}")
    
    # 4. 创建模拟输入数据
    mock_image = torch.randn(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE).to(device)
    mock_points = torch.randint(0, IMG_SIZE, (BATCH_SIZE, 1, 2), device=device).float()
    mock_labels = torch.ones(BATCH_SIZE, 1, device=device).long()
    mock_target_mask = torch.randint(0, NUM_CLASSES, (BATCH_SIZE, IMG_SIZE, IMG_SIZE), device=device).long()

    print("\n--- 输入数据尺寸 ---")
    print(f"图像 (Image):        {mock_image.shape}")
    print(f"点坐标 (Points):     {mock_points.shape}")
    print(f"点标签 (Labels):     {mock_labels.shape}")
    print(f"目标掩码 (Target Mask): {mock_target_mask.shape}")

    # 5. 执行前向和后向传播
    try:
        # --- 前向传播 ---
        predicted_masks = model(mock_image, mock_points, mock_labels)
        print("\n--- 前向传播成功 ---")
        print(f"输出掩码 (Predicted Masks) 尺寸: {predicted_masks.shape}")
        
        expected_shape = (BATCH_SIZE, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
        assert predicted_masks.shape == expected_shape, "输出尺寸错误！"

        # --- 计算损失 ---
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predicted_masks, mock_target_mask)
        print(f"计算损失 (Loss): {loss.item():.4f}")

        # --- 后向传播 ---
        loss.backward()
        print("\n--- 后向传播成功 ---")

        # --- 验证梯度 ---
        grad_check_param = model.image_encoder.patch_embed.proj.weight.grad
        assert grad_check_param is not None, "梯度为 None！后向传播失败。"
        assert grad_check_param.abs().sum() > 0, "梯度全为0！后向传播可能存在问题。"
        print("梯度已成功计算并回传至模型参数。")
        
        print("\n✅ 测试通过: 前向和后向传播均正常工作！")

    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: 传播过程中发生错误。")
        print(f"错误信息: {e}")
        traceback.print_exc()

