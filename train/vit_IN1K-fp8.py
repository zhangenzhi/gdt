import torch
import torch.nn as nn
import timm
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# 确保 transformer_engine 已经安装
# pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git

class TEBlockWrapper(nn.Module):
    """
    一个包装器类，用于将 Transformer Engine 的 TransformerLayer 包装成
    与 timm 库中 vision_transformer.Block 接口一致的模块。

    这样做的目的是为了能够直接替换 timm ViT 模型中的 blocks，
    从而利用 Transformer Engine 带来的性能优势（例如 FP8）。
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            # TE 特有的参数可以在这里添加，或者硬编码
            seq_length=None, # 对于 ViT，需要知道序列长度
    ):
        super().__init__()

        # 参数映射：将 timm.Block 的参数名映射到 TE.TransformerLayer 的参数名
        hidden_size = dim
        ffn_hidden_size = int(dim * mlp_ratio)
        
        # TE 的 dropout 参数是统一的，这里我们取一个平均值或选择一个
        # 对于 ViT，通常 proj_drop 和 attn_drop 是一样的
        dropout = (proj_drop + attn_drop) / 2.0

        # 验证激活层和归一化层是否是 TE 支持的类型
        if not issubclass(act_layer, nn.GELU):
            # TE 的 'gelu' 实现是 nn.GELU 的近似
            print(f"Warning: act_layer {act_layer} is not nn.GELU. TE will use its own 'gelu'.")
        
        if not issubclass(norm_layer, nn.LayerNorm):
            raise ValueError(f"Unsupported norm_layer: {norm_layer}. TE requires LayerNorm.")

        # 实例化 Transformer Engine 的 TransformerLayer
        self.te_layer = te.TransformerLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_heads,
            bias=qkv_bias,
            attention_dropout=attn_drop,
            hidden_dropout=proj_drop,
            drop_path_rate=drop_path,
            # 对于 ViT，我们不需要因果掩码
            self_attn_mask_type="no_mask", 
            # 其他 TE 参数可以根据需要进行设置
            params_dtype=torch.bfloat16, # 推荐使用 bfloat16 或 float16
            normalization='LayerNorm',
            activation='gelu',
            sequence_parallel=False,
            seq_length=seq_length,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现与 timm.Block 一致的前向传播接口。
        为了启用 FP8，需要使用 fp8_autocast 上下文管理器。
        """
        # 注意：TE 的 TransformerLayer 输入期望是 (seq_length, batch_size, hidden_size)
        # 而 timm 的 ViT Block 输入是 (batch_size, seq_length, hidden_size)
        # 因此需要进行转置
        x = x.transpose(0, 1)

        # 使用 FP8 autocast 来获得最佳性能
        # 如果不启用 FP8，可以去掉这个上下文管理器
        with te.fp8_autocast(enabled=True):
             x = self.te_layer(x)

        # 将输出转置回 timm 所期望的格式
        x = x.transpose(0, 1)
        return x

def main():
    """
    主函数，用于演示如何替换和使用 TEBlockWrapper。
    """
    # 检查是否有可用的 CUDA 设备
    if not torch.cuda.is_available():
        print("CUDA is not available. Transformer Engine requires a CUDA-enabled GPU.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. 加载一个标准的 timm ViT 模型
    model_name = 'vit_base_patch16_224'
    print(f"Loading timm model: {model_name}")
    model = timm.create_model(model_name, pretrained=False).to(device, dtype=dtype)
    # 设置为训练模式以计算梯度
    model.train() 

    # 打印原始模型第二个 block 的类型
    print(f"Original model.blocks[1] type: {type(model.blocks[1])}")

    # 2. 获取替换所需的参数
    # 从现有的 block 中提取参数，以确保一致性
    original_block = model.blocks[1]
    dim = original_block.attn.qkv.in_features
    num_heads = original_block.attn.num_heads
    mlp_ratio = original_block.mlp.fc1.out_features / original_block.mlp.fc1.in_features
    
    # ViT 输入的序列长度 = patch_embed.num_patches + 1 (for class token)
    seq_length = model.patch_embed.num_patches + 1

    # 3. 创建 TEBlockWrapper 实例
    print("\nCreating TEBlockWrapper instance...")
    te_wrapper_block = TEBlockWrapper(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True, # ViT base 通常有 bias
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        seq_length=seq_length,
    ).to(device, dtype=dtype)

    # 4. 替换模型中的 block
    print("Replacing model.blocks[1] with TEBlockWrapper...")
    model.blocks[1] = te_wrapper_block
    print(f"New model.blocks[1] type: {type(model.blocks[1])}")

    # 5. 创建一个虚拟输入并进行前向和后向传播测试
    batch_size = 4
    input_resolution = 224
    num_classes = model.num_classes
    
    dummy_input = torch.randn(
        batch_size, 3, input_resolution, input_resolution, 
        device=device, dtype=dtype
    )
    # 为损失计算创建一个虚拟目标
    dummy_target = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    print(f"\nTesting forward and backward pass with input shape: {dummy_input.shape}")
    try:
        # 前向传播
        output = model(dummy_input)
        print("Forward pass successful!")
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, num_classes)

        # 计算损失
        loss = criterion(output, dummy_target)
        print(f"Loss calculated: {loss.item()}")

        # 后向传播
        loss.backward()
        print("Backward pass successful!")

        # 可选：检查 TE 块中某个参数的梯度是否存在
        grad_exists = model.blocks[1].te_layer.self_attention.query_key_value.weight.grad is not None
        print(f"Gradient exists for a parameter in the TE block: {grad_exists}")
        assert grad_exists

    except Exception as e:
        print(f"An error occurred during the forward/backward pass: {e}")

if __name__ == '__main__':
    main()
