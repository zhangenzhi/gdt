import os
import sys
import yaml
import logging
import argparse
import contextlib
from typing import Dict, Any
from functools import partial

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP


import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup
from timm.models.vision_transformer import Block


# 导入 Transformer Engine
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe


# --- 模型创建和替换 ---

def create_timm_vit(config: Dict[str, Any]) -> nn.Module:
    """使用 timm.create_model 创建一个标准的 Vision Transformer 模型。"""
    model_config = config['model']
    model = timm.create_model(
        model_name=model_config['name'],
        pretrained=model_config.get('pretrained', False),
        num_classes=model_config.get('num_classes', 1000),
        img_size=model_config.get('img_size', 224),
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
    )
    return model

def create_timm_vit_with_te_blocks(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    创建一个 timm ViT 模型，并将其 Transformer Blocks 替换为 Transformer Engine 的版本。
    """
    if dist.get_rank() == 0:
        logging.info("正在创建 timm ViT 模型...")
    model = create_timm_vit(config)
    
    if dist.get_rank() == 0:
        logging.info("正在将 timm Blocks 替换为 Transformer Engine Blocks...")

    # 遍历并替换模型中的每一个 Block
    for i, timm_block in enumerate(model.blocks):
        # 从 timm block 获取配置参数
        hidden_size = timm_block.attn.proj.in_features
        num_heads = timm_block.attn.num_heads
        mlp_hidden_size = timm_block.mlp.fc1.out_features

        # 创建一个 Transformer Engine (TE) block
        te_block = te.TransformerLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=mlp_hidden_size,
            num_attention_heads=num_heads,
            # 确保 TE block 的参数与 timm block 匹配
            bias=True, 
            activation='gelu',
            attention_dropout=timm_block.attn.attn_drop.p,
            hidden_dropout=timm_block.attn.proj_drop.p,
            # TE 使用融合的 LayerNorm
            layernorm_epsilon=timm_block.norm1.eps, 
            # 将 TE block 初始化为与 timm 相同的权重
            init_method=lambda w: w, 
        )

        # --- 权重复制 ---
        # 这是一个关键步骤，确保预训练的权重被保留
        with torch.no_grad():
            # 复制 LayerNorm 1 (Attention前的LN)
            te_block.self_attention.layernorm_qkv.weight.copy_(timm_block.norm1.weight)
            te_block.self_attention.layernorm_qkv.bias.copy_(timm_block.norm1.bias)
            
            # 复制 QKV 权重和偏置
            # timm 将 Q, K, V 权重合并在一个张量中，TE 也期望如此
            te_block.self_attention.query_key_value.weight.copy_(timm_block.attn.qkv.weight)
            te_block.self_attention.query_key_value.bias.copy_(timm_block.attn.qkv.bias)
            
            # 复制 Attention Proj 权重和偏置
            te_block.self_attention.proj.weight.copy_(timm_block.attn.proj.weight)
            te_block.self_attention.proj.bias.copy_(timm_block.attn.proj.bias)
            
            # 复制 LayerNorm 2 (MLP前的LN)
            te_block.layernorm_mlp.weight.copy_(timm_block.norm2.weight)
            te_block.layernorm_mlp.bias.copy_(timm_block.norm2.bias)
            
            # 复制 MLP (fc1 和 fc2)
            te_block.mlp.fc1.weight.copy_(timm_block.mlp.fc1.weight)
            te_block.mlp.fc1.bias.copy_(timm_block.mlp.fc1.bias)
            te_block.mlp.fc2.weight.copy_(timm_block.mlp.fc2.weight)
            te_block.mlp.fc2.bias.copy_(timm_block.mlp.fc2.bias)

        # 用新的 TE block 替换旧的 timm block
        model.blocks[i] = te_block.to(device)

    if dist.get_rank() == 0:
        logging.info("所有 timm Blocks 已成功替换。")
    return model

# --- 核心代码 (与之前版本类似，但更加通用) ---

def setup_logging(args):
    """配置日志记录器。"""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "out.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, args, start_epoch, best_val_acc):
    """主训练和评估循环。"""
    num_epochs = config['training']['num_epochs']
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = dist.get_rank() == 0

    fp8_recipe_obj = te_recipe.fp8_recipe(fp8_format=te_recipe.Format.E4M3, amax_history_len=16, amax_compute_algo="max")
    
    if is_main_process:
        logging.info(f"启动训练，使用 {'Transformer Engine FP8' if args.use_fp8 else 'BF16'}...")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
        
    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            sync_context = model.no_sync() if is_accumulation_step else contextlib.nullcontext()
            
            with sync_context:
                autocast_ctx = te.fp8_autocast(enabled=args.use_fp8, fp8_recipe=fp8_recipe_obj) if args.use_fp8 \
                    else torch.cuda.amp.autocast(dtype=torch.bfloat16)

                with autocast_ctx:
                    outputs = model(images)
                
                loss = criterion(outputs.float(), labels)
                loss = loss / accumulation_steps
            
            loss.backward()

            if not is_accumulation_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
                
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 50 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 50
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Acc: {train_acc:.2f}%, LR: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        val_acc = evaluate_model(model, val_loader, device, args)
            
        if is_main_process:
            # ... (检查点保存逻辑与之前相同) ...
            pass

    if is_main_process:
        logging.info(f'训练完成。最佳验证精度: {best_val_acc:.4f}')

def evaluate_model(model, val_loader, device, args):
    """评估函数。"""
    model.eval()
    correct, total = 0, 0
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    if args.use_fp8:
        fp8_recipe_obj = te_recipe.fp8_recipe(fp8_format=te_recipe.Format.E4M3)
        autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_obj)

    with torch.no_grad(), autocast_ctx:
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    total_tensor = torch.tensor(total, device=device)
    correct_tensor = torch.tensor(correct, device=device)
    dist.all_reduce(total_tensor)
    dist.all_reduce(correct_tensor)
    total, correct = total_tensor.item(), correct_tensor.item()
    return 100 * correct / total if total > 0 else 0

def main_worker(args, config):
    """主工作函数，由每个 DDP 进程执行。"""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if rank == 0:
        setup_logging(args)
        logging.info(f"开始训练，共 {world_size} 个进程...")

    # --- 数据加载 (使用timm) ---
    # ... (此处应添加真实的timm数据加载逻辑) ...
    # 为了演示，我们暂时保留模拟数据
    train_dataset = torch.utils.data.TensorDataset(torch.randn(1024, 3, 224, 224), torch.randint(0, 1000, (1024,)))
    val_dataset = torch.utils.data.TensorDataset(torch.randn(256, 3, 224, 224), torch.randint(0, 1000, (256,)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], sampler=val_sampler)
    dataloaders = {'train': train_loader, 'val': val_loader}
    
    # --- 模型创建和转换 ---
    model = create_timm_vit_with_te_blocks(config, device)
    model = DDP(model, device_ids=[local_rank])

    # --- 优化器和调度器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloaders['train']) * config['training']['num_epochs'])
    
    # --- 启动训练循环 ---
    train_loop(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=0, best_val_acc=0.0)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT FP8/BF16 Training Script with timm and TE")
    parser.add_argument('--config', type=str, default='./configs/vit-b16_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='vit-fp8-timm-run', help='Subdirectory for logs/models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to dataset')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--reload', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--use_fp8', action='store_true', help='Use Transformer Engine FP8 for training.')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output = os.path.join(args.output, "imagenet")
    os.makedirs(args.output, exist_ok=True)
    
    main_worker(args, config)