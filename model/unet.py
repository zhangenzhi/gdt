import timm
import torch

def create_unet_model(model_name='unet_efficientnet_b4', pretrained=True, in_chans=3):
    """
    使用timm库创建一个U-Net模型。

    Args:
        model_name (str): timm中U-Net模型的名称。
        pretrained (bool): 是否加载在ImageNet上预训练的encoder权重。
        in_chans (int): 输入图像的通道数 (例如，1代表灰度图，3代表RGB)。

    Returns:
        torch.nn.Module: U-Net模型实例。
    """
    # 如果输入是单通道，预训练的权重（通常用于3通道）将无法加载到第一层卷积。
    # timm会自动处理这个问题，只加载编码器中兼容的部分。
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=1       # 输出为单通道的二值蒙版
    )
    return model

# --- 本地测试 ---
if __name__ == '__main__':
    # 测试创建单通道模型
    model_1_chan = create_unet_model(in_chans=1)
    print(f"成功创建单通道模型: {model_1_chan.__class__.__name__}")
    
    dummy_input = torch.randn(2, 1, 1024, 1024) # (batch, channels=1, height, width)
    output = model_1_chan(dummy_input)
    
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}") # 应为 (2, 1, 1024, 1024)

