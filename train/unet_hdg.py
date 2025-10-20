import os
import sys
sys.path.append("./")

import yaml
import logging
import argparse
import contextlib
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

# --- 导入我们自己的模块 ---
from model.unet import create_unet_model
from model.losses import DiceBCELoss
from dataset.hdg import create_dataloaders

def setup_logging(output_dir, save_dir):
    """配置日志记录到文件和控制台。"""
    log_dir = os.path.join(output_dir, save_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # 清除所有现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_dice_score(logits, targets, smooth=1e-6):
    """计算Dice系数 metric。"""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def visualize_predictions(epoch, output_dir, save_dir, images, targets, preds, val_loader):
    """可视化验证集上的预测结果并保存为图片。"""
    # 反归一化
    mean, std = val_loader.dataset.mean, val_loader.dataset.std
    
    fig, axes = plt.subplots(3, 7, figsize=(28, 12))
    fig.suptitle(f'Epoch {epoch + 1} Validation Results', fontsize=20)
    
    for i in range(7):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * std) + mean # 反归一化
        img = img.clip(0, 1)

        tgt = targets[i].cpu().numpy().squeeze()
        prd = preds[i].cpu().numpy().squeeze()

        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(tgt, cmap='gray')
        axes[1, i].set_title(f'Ground Truth')
        axes[1, i].axis('off')

        axes[2, i].imshow(prd, cmap='gray')
        axes[2, i].set_title(f'Prediction')
        axes[2, i].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, save_dir, f"validation_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    logging.info(f"验证结果可视化已保存至: {save_path}")


def evaluate(model, val_loader, criterion, device, epoch, args):
    """评估模型并可视化结果。"""
    model.eval()
    total_val_loss = 0.0
    total_dice_score = 0.0
    
    # 用于可视化的容器
    all_images, all_targets, all_preds = [], [], []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(dist.get_rank() != 0)):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets)
            
            total_val_loss += loss.item()
            total_dice_score += get_dice_score(logits, targets).item()

            if dist.get_rank() == 0: # 仅在主进程收集可视化数据
                all_images.append(images)
                all_targets.append(targets)
                all_preds.append((torch.sigmoid(logits) > 0.5).float())

    avg_val_loss = total_val_loss / len(val_loader)
    avg_dice_score = total_dice_score / len(val_loader)

    # 在DDP中同步所有进程的评估结果
    metrics_tensor = torch.tensor([avg_val_loss, avg_dice_score], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
    
    synced_loss, synced_dice = metrics_tensor.tolist()

    if dist.get_rank() == 0:
        visualize_predictions(
            epoch, args.output, args.savefile,
            torch.cat(all_images), torch.cat(all_targets), torch.cat(all_preds),
            val_loader
        )

    return synced_loss, synced_dice


def train(config, args):
    """主训练函数"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')

    is_main_process = (local_rank == 0)
    if is_main_process:
        setup_logging(args.output, args.savefile)
        logging.info(f"开始训练，使用 {world_size} 个进程，设备类型: {device.type}")

    # --- 数据加载 ---
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        img_size=config['data']['img_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        use_ddp=True
    )
    # 将计算出的mean和std附加到val_loader.dataset以供可视化使用
    val_loader.dataset.mean, val_loader.dataset.std = train_loader.dataset.mean, train_loader.dataset.std

    # --- 模型创建 ---
    if is_main_process: logging.info(f"正在创建模型: U-Net with {config['model']['backbone_name']} backbone")
    model = create_unet_model(
        backbone_name=config['model']['backbone_name'],
        pretrained=config['model']['pretrained'],
        in_chans=config['model']['in_chans']
    ).to(device)

    if config['training'].get('use_compile', False):
        if is_main_process: logging.info("正在应用 torch.compile()...")
        model = torch.compile(model)
        
    model = DDP(model, device_ids=[local_rank])

    # --- 损失，优化器，调度器 ---
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    optimizer_steps_per_epoch = len(train_loader) // accumulation_steps
    num_training_steps = num_epochs * optimizer_steps_per_epoch
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch
    
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    scaler = GradScaler(enabled=True)

    # --- 检查点加载 ---
    start_epoch = 0
    best_dice = 0.0
    checkpoint_dir = os.path.join(args.output, args.savefile)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if args.reload and os.path.exists(latest_checkpoint_path):
        if is_main_process: logging.info(f"从检查点恢复训练: {latest_checkpoint_path}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        if is_main_process: logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。")

    # --- 训练循环 ---
    if is_main_process:
        logging.info(f"开始训练，总共 {num_epochs} 个 epochs...")
        logging.info(f"配置: {config}")

    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(not is_main_process))):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            is_sync_step = (i + 1) % accumulation_steps == 0
            sync_context = contextlib.nullcontext() if is_sync_step else model.no_sync()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits, targets)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
            
            if is_sync_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- 评估 ---
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, epoch, args)
        
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")
            
            # 保存最新检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dice': best_dice,
            }, latest_checkpoint_path)

            # 如果是最佳模型，则另外保存一份
            if val_dice > best_dice:
                best_dice = val_dice
                logging.info(f"发现新的最佳模型，Dice: {best_dice:.4f}. 正在保存...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.module.state_dict(), best_checkpoint_path)

    if is_main_process:
        logging.info(f'训练完成. 最佳验证Dice系数: {best_dice:.4f}')
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net Hydrogel Segmentation Training Script")
    parser.add_argument('--config', type=str, default='./configs/unet-resnet_HDG.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='/work/c30636/dataset/hydrogel-s', help='数据集目录路径 (覆盖配置文件中的设置)')
    parser.add_argument('--output', type=str, default='./output', help='输出根目录 (覆盖配置文件中的设置)')
    parser.add_argument('--savefile', type='unet_resnet18_1k', default='unet_resnet18_1k', help='本次运行的保存文件夹名 (覆盖配置文件中的设置)')
    parser.add_argument('--reload', action='store_true', help='从最新的检查点恢复训练')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 命令行参数优先
    if args.data_dir: config['data']['data_dir'] = args.data_dir
    if args.output: config['output']['base_dir'] = args.output
    if args.savefile: config['output']['save_dir'] = args.savefile
        
    args.data_dir = config['data']['data_dir']
    args.output = config['output']['base_dir']
    args.savefile = config['output']['save_dir']

    train(config, args)
