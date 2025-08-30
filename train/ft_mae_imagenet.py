import os
import sys
import yaml
import logging
import argparse
import contextlib
from collections import OrderedDict

import torch
from torch import nn
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy # 修正：使用支持 Mixup 的损失函数

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast 
import timm # 确保导入 timm

# Import the baseline ViT model and the dataset functions
sys.path.append("./")
from dataset.imagenet import imagenet_distribute
from dataset.utlis import param_groups_lrd

# --------------------------------------------- #
#   1. 日志与 DDP 设置 (保持不变)
# --------------------------------------------- #
def setup_logging(args):
    """配置日志记录到文件和控制台。"""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

# --------------------------------------------- #
#   2. 新增：MAE 权重加载逻辑
# --------------------------------------------- #
def load_mae_checkpoint(model, checkpoint_path):
    """
    从 MAE 预训练检查点加载 encoder 权重到 timm ViT 模型。
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # MAE 检查点通常包含 'model_state_dict' 或 'model'
    if 'model_state_dict' in checkpoint:
        mae_state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        mae_state_dict = checkpoint['model']
    else:
        raise KeyError("在检查点中找不到 'model_state_dict' 或 'model'。")

    new_state_dict = OrderedDict()
    # 遍历 MAE 的 state_dict，只提取 encoder 的权重
    for key, value in mae_state_dict.items():
        # 移除 'encoder.model.' 或 'encoder.' 前缀
        if key.startswith('encoder.model.'):
            new_key = key.replace('encoder.model.', '')
            new_state_dict[new_key] = value
        elif key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
            new_state_dict[new_key] = value
            
    # 加载权重，strict=False 允许我们只加载骨干网络部分
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    logging.info(f"成功从 MAE 检查点恢复权重: {checkpoint_path}")
    logging.warning(f"权重加载信息: {msg}")
    
    # 断言确保丢失的只有分类头，这是正常现象
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}, \
        f"加载权重时丢失了非预期的键: {msg.missing_keys}"
    logging.info("骨干网络权重加载成功，分类头将随机初始化。")

def get_finetune_model(config, args):
    """
    创建一个 timm ViT 模型，并从 MAE 检查点加载预训练权重。
    """
    model = timm.create_model(
        config['model']['name'],
        pretrained=False, # 我们将手动加载权重
        num_classes=config['model']['num_classes'],
        drop_path_rate=config['model'].get('drop_path_rate', 0.1)
    )
    
    # 加载 MAE 检查点
    load_mae_checkpoint(model, args.mae_checkpoint)
    
    return model

# --------------------------------------------- #
#   3. 训练与评估循环 (适配和优化)
# --------------------------------------------- #
def train_vit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device_id, args, start_epoch, best_val_acc=0.0, is_ddp=False):
    scaler = GradScaler(enabled=True)
    num_epochs = config['training']['num_epochs']
    
    mixup_fn = Mixup(
        mixup_alpha=config['data'].get('mixup', 0.8),
        cutmix_alpha=config['data'].get('cutmix', 1.0),
        label_smoothing=config['data'].get('label_smoothing', 0.1),
        prob=config['data'].get('mixup_prob', 1.0),
        num_classes=config['model']['num_classes']
    )
    
    is_main_process = not is_ddp or (dist.get_rank() == 0)

    if is_main_process:
        logging.info(f"开始在 ImageNet-1K 上微调 MAE Encoder...")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            # 保存原始标签用于计算准确率
            original_labels = labels.clone()
            
            # 应用 Mixup/CutMix
            images, soft_labels = mixup_fn(images, labels)
                
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, soft_labels) # 使用软标签计算损失
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # --- 评估逻辑 ---
            _, predicted = torch.max(outputs.data, 1)
            running_total += original_labels.size(0)
            running_corrects += (predicted == original_labels).sum().item()
            running_loss += loss.item()

            if (i + 1) % 50 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 50
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        # 每个 epoch 结束后更新学习率
        if scheduler:
            scheduler.step()

        val_acc = evaluate_model_compatible(model, val_loader, device_id, is_ddp=is_ddp)
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f}% | LR: {current_lr:.6f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"新的最佳验证精度: {best_val_acc:.4f}%. 保存最佳模型...")
                checkpoint_dir = os.path.join(args.output, args.savefile)
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_acc': best_val_acc,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'训练完成。最佳验证精度: {best_val_acc:.4f}%')

def evaluate_model_compatible(model, val_loader, device, is_ddp=False):
    """评估函数。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    if is_ddp:
        total_tensor = torch.tensor(total, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        dist.all_reduce(total_tensor)
        dist.all_reduce(correct_tensor)
        total, correct = total_tensor.item(), correct_tensor.item()
        
    return 100 * correct / total if total > 0 else 0

# --------------------------------------------- #
#   4. 主函数 (集成新逻辑)
# --------------------------------------------- #
def vit_imagenet_train_single(args, config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"开始微调，使用 {world_size} 个进程。")

    # --- 数据加载器 (保持不变) ---
    dataloaders = imagenet_distribute(
        img_size=config['data']['img_size'],
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # --- 模型创建与权重加载 ---
    model = get_finetune_model(config, args).to(device)
        
    if config['training'].get('use_compile', False):
        if dist.get_rank() == 0: logging.info("正在应用 torch.compile()...")
        model = torch.compile(model)
        
    model = DDP(model, device_ids=[local_rank])

    # --- 损失函数、优化器、调度器 ---
    criterion = SoftTargetCrossEntropy() # 修正：使用 SoftTargetCrossEntropy
    
    model_without_ddp = model.module
    param_groups = param_groups_lrd(
        model_without_ddp, 
        config['training']['weight_decay'],
        layer_decay=config['training']['layer_decay']
    )
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', (0.9, 0.95)))
    )
    
    training_config = config['training']
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 0)
    
    # 保持原有的学习率调度器逻辑
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs * len(dataloaders['train']))
        main_scheduler = CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs) * len(dataloaders['train']), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs * len(dataloaders['train'])])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloaders['train']), eta_min=1e-6)
    
    start_epoch, best_val_acc = 0, 0.0
    # ... 检查点恢复逻辑 (保持不变) ...
            
    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=(world_size > 1))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Fine-tuning Script with MAE Encoder")
    parser.add_argument('--task', type=str, default='mae_finetune', help='Type of task')
    parser.add_argument('--config', type=str, default='./configs/ft_mae_vit-b16_IN1K.yaml', help='Path to the YAML configuration file (ft_vit-b16_IN1K.yaml).')
    # 新增：指定 MAE 检查点路径的参数
    parser.add_argument('--mae_checkpoint', type=str, default='./output/mae_pretrain/mae_vit-b16-timm/best_model.pth', help='Path to the pre-trained MAE checkpoint.')
    parser.add_argument('--output', type=str, default='./output/finetune', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='mae-ft-vit-b-16', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    
    args = parser.parse_args()

    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update args from config for consistency
    args.img_size = config['data']['img_size']
    args.num_epochs = config['training']['num_epochs']
    args.batch_size = config['training']['batch_size']
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    vit_imagenet_train_single(args, config)
