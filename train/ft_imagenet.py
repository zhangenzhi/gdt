import os
import sys
import yaml
import logging
import argparse
import contextlib
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

import timm
from timm.data import create_transform, Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer

# 确保可以从项目根目录导入
sys.path.append("./")
from dataset.imagenet import build_imagenet_dataset # 假设你有这个数据集构建函数

# --------------------------------------------- #
#   1. 日志和分布式训练设置
# --------------------------------------------- #
def setup_logging(args):
    """配置日志记录到文件和控制台。"""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 清除已有处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "finetune_out.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --------------------------------------------- #
#   2. 模型定义与权重加载
# --------------------------------------------- #
def load_mae_checkpoint(model: VisionTransformer, checkpoint_path: str):
    """
    从 MAE 预训练的检查点加载权重到 timm 的 ViT 模型。
    这个函数会处理 state_dict 中键名的不匹配问题。
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    mae_state_dict = checkpoint['model_state_dict']
    
    # 创建一个新的 state_dict 来存储重映射后的键
    new_state_dict = OrderedDict()
    
    # 遍历 MAE 的 state_dict
    for key, value in mae_state_dict.items():
        # 我们只需要 encoder 的权重，并移除 'encoder.model.' 前缀
        if key.startswith('encoder.model.'):
            new_key = key.replace('encoder.model.', '')
            new_state_dict[new_key] = value
            
    # 加载权重到 ViT 模型
    # strict=False 允许我们只加载匹配的键（即骨干网络部分）
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    # 打印加载信息，检查是否有未匹配的键
    logging.info(f"成功从 MAE 检查点恢复权重: {checkpoint_path}")
    logging.warning(f"权重加载信息: {msg}")
    
    # 检查丢失的键是否只有 head.weight 和 head.bias，这是正常的
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}, \
        "加载权重时丢失了非预期的键！"
    logging.info("成功加载骨干网络权重，分类头将随机初始化。")

def get_finetune_model(args, config):
    """创建 ViT 模型并加载 MAE 预训练权重。"""
    model = timm.create_model(
        config['model']['name'],
        pretrained=False, # 我们将手动加载权重
        num_classes=config['model']['num_classes'],
        drop_path_rate=config['model']['drop_path_rate']
    )
    
    # 加载 MAE 检查点
    load_mae_checkpoint(model, args.mae_checkpoint)
    
    return model

# --------------------------------------------- #
#   3. 层衰减学习率 (LLRD)
# --------------------------------------------- #
def get_llrd_param_groups(model: VisionTransformer, lr, weight_decay, layer_decay):
    """为 ViT 模型参数分组以应用层衰减学习率。"""
    param_groups = {}
    
    # 遍历所有命名参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 根据参数名称判断其所在的层深度
        if name.startswith('cls_token') or name.startswith('pos_embed'):
            depth = 0
        elif name.startswith('patch_embed'):
            depth = 0
        elif name.startswith('blocks.'):
            depth = int(name.split('.')[1]) + 1
        else: # 分类头
            depth = model.get_classifier().depth # timm ViT 通常是12层
        
        # 计算该层的学习率
        group_lr = lr * (layer_decay ** (model.get_classifier().depth - depth))
        
        group_name = f'layer_{depth}_lr_{group_lr:.6f}'
        
        if group_name not in param_groups:
            param_groups[group_name] = {
                'lr': group_lr,
                'params': [],
                'weight_decay': weight_decay
            }
            
        param_groups[group_name]['params'].append(param)
        
    # 将字典转换为优化器可接受的列表格式
    return list(param_groups.values())

# --------------------------------------------- #
#   4. 训练与评估循环
# --------------------------------------------- #
def train_one_epoch(model, loader, optimizer, loss_fn, mixup_fn, device, epoch, is_main_process):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # 应用 Mixup/CutMix
        images, targets = mixup_fn(images, targets)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        # 计算准确率
        preds = outputs.argmax(dim=1)
        ground_truth = targets.argmax(dim=1) # Mixup 后的 target 是 one-hot
        total_correct += (preds == ground_truth).sum().item()
        total_samples += images.size(0)
        
        if (i + 1) % 50 == 0 and is_main_process:
            logging.info(f'[Epoch {epoch+1}, Batch {i+1}] Train Loss: {loss.item():.4f}, Train Acc: {(total_correct/total_samples)*100:.2f}%')
            
    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
            
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)
            
    return total_correct / total_samples

# --------------------------------------------- #
#   5. 主函数
# --------------------------------------------- #
def main(args, config):
    # DDP 设置
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = world_size > 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if is_ddp:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    is_main_process = not is_ddp or (local_rank == 0)
    if is_main_process:
        setup_logging(args)
        logging.info(f"开始在 ImageNet-1K 上微调，使用 {world_size} 个进程。")

    # --- 数据加载 ---
    train_transform = create_transform(
        input_size=config['data']['img_size'],
        is_training=True,
        color_jitter=config['data']['color_jitter'],
        auto_augment=config['data']['auto_augment'],
        re_prob=config['data']['re_prob'],
        re_mode=config['data']['re_mode'],
        re_count=config['data']['re_count'],
    )
    val_transform = create_transform(
        input_size=config['data']['img_size'],
        is_training=False
    )
    
    train_dataset = build_imagenet_dataset(is_train=True, data_dir=args.data_dir, transform=train_transform)
    val_dataset = build_imagenet_dataset(is_train=False, data_dir=args.data_dir, transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    
    # --- 模型、优化器、损失函数 ---
    model = get_finetune_model(args, config).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # Mixup
    mixup_fn = Mixup(
        mixup_alpha=config['data']['mixup'],
        cutmix_alpha=config['data']['cutmix'],
        prob=config['data']['mixup_prob'],
        num_classes=config['model']['num_classes']
    )
    
    # 优化器参数分组 (LLRD)
    param_groups = get_llrd_param_groups(
        model.module if is_ddp else model,
        config['training']['learning_rate'],
        config['training']['weight_decay'],
        config['training']['layer_decay']
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    # 损失函数
    loss_fn = SoftTargetCrossEntropy()

    # 学习率调度器
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training']['warmup_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    
    # --- 训练循环 ---
    best_acc = 0.0
    if is_main_process:
        logging.info(f"开始微调 {num_epochs} 个 epochs...")
        
    for epoch in range(num_epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, mixup_fn, device, epoch, is_main_process)
        
        # 更新学习率
        if epoch >= warmup_epochs:
            scheduler.step()
            
        val_acc = evaluate(model, val_loader, device)
        
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc*100:.2f}% | Train Acc: {train_acc*100:.2f}% | LR: {current_lr:.6f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                logging.info(f"新的最佳验证准确率: {best_acc*100:.2f}%. 保存模型...")
                
                checkpoint_dir = os.path.join(args.output, args.savefile)
                best_checkpoint_path = os.path.join(checkpoint_dir, "finetune_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': (model.module if is_ddp else model).state_dict(),
                    'best_acc': best_acc,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'微调完成。最佳验证准确率: {best_acc*100:.2f}%')
        
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE Fine-tuning Script on ImageNet")
    parser.add_argument('--config', type=str, default='./configs/ft_vit-b16_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--mae_checkpoint', type=str, default='./output/mae_pretrain/mae_vit-b16-timm/, help='Path to the pre-trained MAE checkpoint.')
    parser.add_argument('--output', type=str, default='./output/finetune', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='mae_vit-b16_finetune', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/path/to/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs(args.output, exist_ok=True)
    main(args, config)
