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
        
        # U-Net的瓶颈部分 (通常是编码器最深层的输出)
        # 我们直接使用编码器的最深层特征，不再额外加bottleneck层
        
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
        # `timm`的ResNet backbone第一层stride=2，所以最终输出需要额外一次上采样
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = nn.Conv2d(channels_from_below, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器提取特征
        features = self.backbone(x)
        features.reverse()
        
        x_decoder = features[0]
        for i, up_layer in enumerate(self.up_layers):
            skip_connection = features[i + 1]
            x_decoder = up_layer(x_decoder, skip_connection)
        
        # 最终上采样和输出
        x_final = self.final_up(x_decoder)
        logits = self.outc(x_final)
        
        return logits

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
        out_indices=(1, 2, 3, 4), # 指定输出4个阶段的特征图
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