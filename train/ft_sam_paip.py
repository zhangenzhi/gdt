import os
import sys
import yaml
import logging
import argparse
from collections import OrderedDict

import tifffile
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast 

sys.path.append("./")
from model.sam import SAMLikeModel
from dataset.s8d import build_s8d_segmentation_dataloaders

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
            logging.FileHandler(os.path.join(log_dir, "finetune_seg_out.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

class DiceLoss(nn.Module):
    """多类别分割的 Dice Loss"""
    def __init__(self, num_classes, softmax_dim=1, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim
        self.smooth = smooth

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=self.softmax_dim)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        intersection = torch.sum(probas * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probas + targets_one_hot, dim=(2, 3))
        
        dice_score = ((2. * intersection + self.smooth) / (cardinality + self.smooth))
        dice_loss = 1 - dice_score.mean()
        return dice_loss

def calculate_dice_score(pred, target, num_classes, smooth=1e-5):
    """计算平均 Dice Score 作为评估指标"""
    dice_list = []
    pred = torch.argmax(pred, dim=1)
    
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        
        if union == 0:
            dice = 1.0
        else:
            dice = (2. * intersection + smooth) / (union + smooth)
        dice_list.append(dice)
            
    mean_dice = torch.tensor(dice_list).mean().item()
    return mean_dice * 100


def train_segmentation_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device_id, args, start_epoch, best_val_dice=0.0, is_ddp=False):
    scaler = GradScaler(enabled=True)
    num_epochs = config['training']['num_epochs']
    is_main_process = not is_ddp or (dist.get_rank() == 0)

    if is_main_process:
        logging.info(f"开始在 S8D 上微调分割模型 (无提示)...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_dice = 0.0, 0.0
        
        # 关键修改：解包三个返回值
        for i, (images, masks, _) in enumerate(train_loader):
            images = images.to(device_id, non_blocking=True)
            masks = masks.to(device_id, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
            
            running_loss += loss.item()
            with torch.no_grad():
                dice = calculate_dice_score(logits.cpu(), masks.cpu(), config['model']['num_classes'])
                running_dice += dice

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                avg_dice = running_dice / 10
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.2f}%')
                running_loss, running_dice = 0.0, 0.0

        val_dice = evaluate_segmentation_model(model, val_loader, device_id, config['model']['num_classes'], is_ddp=is_ddp)
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Dice Score: {val_dice:.2f}% | LR: {current_lr:.6f}")
            
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                logging.info(f"新的最佳验证 Dice Score: {best_val_dice:.2f}%. 保存最佳模型...")
                checkpoint_dir = os.path.join(args.output, args.savefile)
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'best_val_dice': best_val_dice,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'训练完成。最佳验证 Dice Score: {best_val_dice:.2f}%')

def evaluate_segmentation_model(model, val_loader, device, num_classes, is_ddp=False):
    """评估分割模型的 Dice Score"""
    model.eval()
    total_dice, num_samples = 0.0, 0
    with torch.no_grad():
        # 关键修改：解包三个返回值
        for images, masks, _ in val_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(images)
            
            total_dice += calculate_dice_score(logits.cpu(), masks.cpu(), num_classes) * images.size(0)
            num_samples += images.size(0)

    if is_ddp:
        total_dice_tensor = torch.tensor(total_dice, device=device)
        num_samples_tensor = torch.tensor(num_samples, device=device)
        dist.all_reduce(total_dice_tensor)
        dist.all_reduce(num_samples_tensor)
        total_dice, num_samples = total_dice_tensor.item(), num_samples_tensor.item()
        
    return total_dice / num_samples if num_samples > 0 else 0

def segmentation_s8d_finetune_ddp(args, config, mae_config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)

    dataloaders = build_s8d_segmentation_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )
    
    model = SAMLikeModel(config=mae_config, num_classes=config['model']['num_classes']).to(device)
    model.load_vit_backbone(args.mae_checkpoint)
        
    if config['training'].get('use_compile', False):
        if dist.get_rank() == 0: logging.info("正在应用 torch.compile()...")
        model = torch.compile(model)
        
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    encoder_params = [p for n, p in model.module.image_encoder.named_parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.module.named_parameters() if not n.startswith('image_encoder.') and p.requires_grad]
    
    param_groups = [
        {'params': encoder_params, 'lr': config['training']['encoder_lr']},
        {'params': decoder_params, 'lr': config['training']['decoder_lr']}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config['training']['weight_decay'])
    criterion = DiceLoss(num_classes=config['model']['num_classes'])
    
    scheduler = None 
    
    train_segmentation_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=0, best_val_dice=0.0, is_ddp=(world_size > 1))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Model Finetuning on S8D")
    parser.add_argument('--task', type=str, default='segmentation_finetune_s8d')
    parser.add_argument('--config', type=str, default='./configs/ft-mae-sam-b_S8D.yaml')
    parser.add_argument('--mae_config', type=str, default='./configs/mae-vit-b_S8D.yaml') 
    parser.add_argument('--mae_checkpoint', type=str, default='./output/mae_pretrain/mae-n64-bz1024-w0/best_model.pth')
    
    parser.add_argument('--output', type=str, default='./output/finetune')
    parser.add_argument('--savefile', type=str, default='seg-ft-sam-s8d')
    # 关键修改：更新了默认的数据目录
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/s8d/finetune/Noise_0.05_Blur_2_sparsity_2_NumAng_3600")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.mae_config, 'r') as f:
        mae_config = yaml.safe_load(f)
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    segmentation_s8d_finetune_ddp(args, config, mae_config)