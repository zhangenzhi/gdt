import timm
import torch
import torch.nn as nn

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
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CustomUNet(nn.Module):
    def __init__(self, encoder, n_classes=1, bilinear=True):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 获取编码器各阶段输出的通道数
        encoder_channels = self.encoder.feature_info.channels()
        
        # U-Net的瓶颈部分 (bottleneck)
        self.bottleneck = DoubleConv(encoder_channels[-1], encoder_channels[-1] * 2)

        # U-Net的解码器/上采样部分 (修正后的逻辑)
        self.up_layers = nn.ModuleList()
        # 获取编码器各阶段输出的通道数，并将其反转以便从深到浅处理
        reversed_encoder_channels = self.encoder.feature_info.channels()[::-1]
        
        # 从瓶颈层开始，这是解码器第一个上采样层的输入
        channels_from_below = encoder_channels[-1] * 2

        # 遍历跳跃连接的通道数 (从深到浅)
        # 我们跳过第一个 (reversed_encoder_channels[0])，因为它用于瓶颈层
        for skip_channels in reversed_encoder_channels[1:]:
            # Up block的输入通道数 = 来自下一层的通道数 + 来自跳跃连接的通道数
            total_in_channels = channels_from_below + skip_channels
            # Up block的输出通道数，通常设置为与跳跃连接的通道数相同
            out_channels = skip_channels
            
            self.up_layers.append(Up(total_in_channels, out_channels, bilinear))
            
            # 更新下一轮循环的"来自下一层的通道数"
            channels_from_below = out_channels

        # 最终输出层，其输入通道数应为最后一个Up block的输出通道数
        self.outc = nn.Conv2d(channels_from_below, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器提取特征
        features = self.encoder(x)
        features.reverse()  # 将特征反转，方便从深到浅处理
        
        # 瓶颈
        x = self.bottleneck(features[0])

        # 解码器和跳跃连接
        for i, up_layer in enumerate(self.up_layers):
            skip_connection = features[i + 1]
            x = up_layer(x, skip_connection)

        logits = self.outc(x)
        return logits

def create_unet_model(encoder_name='efficientnet_b4', pretrained=True, in_chans=1):
    """
    使用timm库创建一个U-Net模型。

    Args:
        encoder_name (str): timm中用作编码器的模型名称。
        pretrained (bool): 是否加载在ImageNet上预训练的encoder权重。
        in_chans (int): 输入图像的通道数 (例如，1代表灰度图，3代表RGB)。

    Returns:
        torch.nn.Module: U-Net模型实例。
    """
    encoder = timm.create_model(
        encoder_name,
        pretrained=pretrained,
        in_chans=in_chans,
        features_only=True,  # <-- 关键：让模型返回中间层的特征图
    )
    model = CustomUNet(encoder, n_classes=1)
    return model

# --- 本地测试 ---
if __name__ == '__main__':
    # 测试创建单通道模型
    # 注意：现在我们传入的是编码器的名称
    model_1_chan = create_unet_model(encoder_name='efficientnet_b4', pretrained=False, in_chans=1)
    print(f"成功创建基于timm encoder的U-Net模型: {model_1_chan.__class__.__name__}")
    
    dummy_input = torch.randn(2, 1, 1024, 1024)
    output = model_1_chan(dummy_input)
    
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}") # 应为 (2, 1, 1024, 1024)

