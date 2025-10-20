import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 保证跳跃连接的特征图尺寸与上采样的特征图尺寸一致
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithBackbone(nn.Module):
    def __init__(self, backbone, n_classes=1, bilinear=True):
        super().__init__()
        self.backbone = backbone
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 获取编码器各阶段输出的通道数
        encoder_channels = self.backbone.feature_info.channels()
        
        # U-Net的解码器/上采样部分
        self.up_layers = nn.ModuleList()
        reversed_encoder_channels = encoder_channels[::-1]
        
        # 从最深层开始，这是解码器第一个上采样层的输入
        channels_from_below = reversed_encoder_channels[0]

        # 遍历跳跃连接的通道数 (从深到浅)
        # 我们跳过第一个 (reversed_encoder_channels[0])
        for skip_channels in reversed_encoder_channels[1:]:
            total_in_channels = channels_from_below + skip_channels
            out_channels = skip_channels
            self.up_layers.append(Up(total_in_channels, out_channels, bilinear))
            channels_from_below = out_channels

        # 最终输出层
        self.outc = nn.Conv2d(channels_from_below, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器提取特征
        features = self.backbone(x)
        features.reverse()
        
        x_decoder = features[0]
        for i, up_layer in enumerate(self.up_layers):
            skip_connection = features[i + 1]
            x_decoder = up_layer(x_decoder, skip_connection)
        
        # 最终输出
        # 先在低分辨率上进行卷积，然后直接上采样到原始输入尺寸，这样更高效且灵活
        logits = self.outc(x_decoder)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

def create_unet_model(backbone_name='resnet34', pretrained=True, in_chans=1):
    """
    使用timm库和指定的backbone创建一个U-Net模型。

    Args:
        backbone_name (str): timm中用作编码器的模型名称 (e.g., 'resnet34', 'resnet50').
        pretrained (bool): 是否加载在ImageNet上预训练的backbone权重。
        in_chans (int): 输入图像的通道数。

    Returns:
        torch.nn.Module: U-Net模型实例。
    """
    backbone = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        in_chans=in_chans,
        features_only=True,
        out_indices=(0, 1, 2, 3, 4), # 指定输出5个阶段的特征图以匹配U-Net结构
    )
    model = UNetWithBackbone(backbone, n_classes=1)
    return model

# --- 本地测试 ---
if __name__ == '__main__':
    print("--- 针对1k图像，测试创建基于ResNet18的U-Net ---")
    model_1k = create_unet_model(backbone_name='resnet18', pretrained=False, in_chans=1)
    print(f"成功创建模型: {model_1k.__class__.__name__} with ResNet18 backbone")
    
    dummy_input_1k = torch.randn(2, 1, 1024, 1024)
    output_1k = model_1k(dummy_input_1k)
    
    print(f"1k 输入尺寸: {dummy_input_1k.shape}")
    print(f"1k 输出尺寸: {output_1k.shape}")
    assert dummy_input_1k.shape[2:] == output_1k.shape[2:], "1k 输出尺寸与输入尺寸不匹配!"
    print("1k 模型结构验证成功。\n")

    print("--- 针对8k图像，测试创建基于ResNet34的U-Net (包括前向和反向传播) ---")
    if not torch.cuda.is_available():
        print("未检测到CUDA GPU，跳过8k显存占用测试。")
    else:
        # 清空1k测试可能占用的缓存，确保8k测试的显存测量更准确
        print("正在清空CUDA缓存...")
        torch.cuda.empty_cache()
        
        device = torch.device("cuda")
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        
        try:
            # 1. 创建模型并移至GPU
            model_8k = create_unet_model(backbone_name='resnet18', pretrained=False, in_chans=1)
            model_8k.to(device)
            print(f"成功创建模型: {model_8k.__class__.__name__} with ResNet34 backbone")

            # 2. 创建输入和目标张量
            # 使用bfloat16以模拟训练环境，节省显存
            dummy_input_8k = torch.randn(1, 1, 8192, 8192, device=device, dtype=torch.bfloat16)
            dummy_target_8k = torch.rand(1, 1, 8192, 8192, device=device, dtype=torch.bfloat16)
            print(f"8k 输入尺寸: {dummy_input_8k.shape}")
            
            # 3. 验证前向传播
            print("正在执行前向传播...")
            torch.cuda.reset_peak_memory_stats(device)
            initial_mem = torch.cuda.memory_allocated(device) / 1024**3
            
            # 使用amp.autocast模拟混合精度训练
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output_8k = model_8k(dummy_input_8k)
                loss = F.mse_loss(output_8k, dummy_target_8k) # 使用简单的损失函数进行测试

            forward_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"前向传播完成。输出尺寸: {output_8k.shape}")
            assert dummy_input_8k.shape[2:] == output_8k.shape[2:], "8k 输出尺寸与输入尺寸不匹配!"
            print(f"初始显存占用: {initial_mem:.2f} GB")
            print(f"前向传播峰值显存占用: {forward_mem:.2f} GB")

            # 4. 验证反向传播
            print("正在执行反向传播...")
            # GradScaler是可选的，但为了完整性我们加上
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            
            backward_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print("反向传播完成。")
            print(f"前向+反向传播峰值显存占用: {backward_mem:.2f} GB")
            print("\n8k 模型结构和显存占用验证成功。")

        except torch.cuda.OutOfMemoryError:
            print("\n错误: 发生CUDA显存不足错误!")
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"峰值显存占用达到 {peak_mem:.2f} GB 时发生错误。")
            print("请考虑使用更轻量的backbone，减小图像尺寸，或使用梯度累积等技术。")
        except Exception as e:
            print(f"\n在测试8k输入时遇到意外错误: {e}")

