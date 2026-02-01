import os
import sys
import yaml
import logging
import argparse
import contextlib

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from matplotlib import pyplot as plt
import numpy as np
import cv2

# --- 项目特定导入 ---
sys.path.append("./")

from model.hvit import HMAEVIT as HVIT
from gdt.hmae import HierarchicalMaskedAutoEncoder
from dataset.s8d_hmae import S8DPretrainDataset

def setup_logging(args):
    """配置日志到文件和控制台。"""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
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

def visualize_hmae_reconstruction(processed_batch, pred_img, loss, step, output_dir, prefix="train"):
    """
    可视化 HMAE 任务的结果。
    显示：原始 Patch (Target), 噪声 Patch (Input), 重建 Patch (Recon)。
    """
    # 获取第一张图的数据
    target_patches = processed_batch['targets'][0].cpu() # [L, C, P, P]
    input_patches = processed_batch['patches'][0].cpu()   # [L, C, P, P]
    mask = processed_batch['mask'][0].cpu()              # [L]
    recon_patches = pred_img[0].cpu()                    # [L, C, P, P]
    
    # 简单的网格可视化逻辑
    L = target_patches.shape[0]
    indices = torch.where(mask == 1)[0][:16] # 选前16个被遮罩的 Patch
    if len(indices) == 0: return

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 2 * len(indices)))
    for i, idx in enumerate(indices):
        # Target
        axes[i, 0].imshow(target_patches[idx].permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title("Target")
        # Input (Noised)
        axes[i, 1].imshow(input_patches[idx].permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title("Input")
        # Recon
        axes[i, 2].imshow(recon_patches[idx].detach().permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 2].set_title("Recon")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}.png"))
    plt.close(fig)

def run_pretrain_batch(batch, model, hmae_engine, device_id):
    """
    辅助函数：处理 batch 原始数据，转换为模型输入，并计算模型前向。
    """
    images = batch['image'] # [B, 1, H, W]
    edges = batch['edges']  # [B, 1, H, W]
    
    # --- 1. 处理器预处理 (CPU 端进行四叉树分解和噪声注入) ---
    batch_patches, batch_targets, batch_noises, batch_coords, batch_depths, batch_masks = [], [], [], [], [], []
    
    # 将 Tensor 转换为 NumPy 进行 CPU 密集型处理
    imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
    edges_np = (edges.squeeze(1).cpu().numpy() * 255).astype('uint8')

    for b in range(images.shape[0]):
        p, t, n, c, d, m = hmae_engine.process_single(imgs_np[b], edges_np[b])
        batch_patches.append(p); batch_targets.append(t); batch_noises.append(n)
        batch_coords.append(c); batch_depths.append(d); batch_masks.append(m)

    # 包装成 dict 方便传递
    processed = {
        'patches': torch.stack(batch_patches).to(device_id, non_blocking=True),
        'targets': torch.stack(batch_targets).to(device_id, non_blocking=True),
        'noises': torch.stack(batch_noises).to(device_id, non_blocking=True),
        'coords': torch.stack(batch_coords).to(device_id, non_blocking=True),
        'depths': torch.stack(batch_depths).to(device_id, non_blocking=True),
        'mask': torch.stack(batch_masks).to(device_id, non_blocking=True)
    }

    # --- 2. 模型推理 ---
    pred_img, pred_noise = model(processed['patches'], processed['coords'], processed['depths'], processed['mask'])
    
    # --- 3. 处理器计算 Loss ---
    loss = hmae_engine.train_step_loss(processed['targets'], pred_img, processed['noises'], pred_noise, processed['mask'])
    
    return loss, pred_img, processed

def pretrain_hmae_model(model, hmae_engine, train_loader, val_loader, optimizer, scheduler, num_epochs, device_id, args, config, start_epoch, is_ddp=False):
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = not is_ddp or (dist.get_rank() == 0)

    if is_main_process:
        logging.info(f"Starting HMAE pre-training for {num_epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 数据交给引擎预处理 -> 模型推理 -> 引擎算 Loss
                    loss, pred_img, processed = run_pretrain_batch(batch, model, hmae_engine, device_id)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            # 定期可视化
            if (i + 1) % 500 == 0 and is_main_process:
                visualize_hmae_reconstruction(
                    processed, pred_img, loss.item(), i + 1,
                    os.path.join(args.output, args.savefile, "images"),
                    prefix=f"train_e{epoch + 1}"
                )

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.5f}, LR: {current_lr:.6f}')
                running_loss = 0.0

        # 每个 Epoch 进行验证 (略，结构相似)
        # val_loss = evaluate_hmae_model(...) 
        
    if is_main_process:
        logging.info('Finished Pre-training.')

def hmae_s8d_pretrain_ddp(args, config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"DDP Initialized. Rank: {local_rank}, World Size: {world_size}")

    # 1. 数据集
    dataset = S8DPretrainDataset(args.data_dir, img_size=config['model']['img_size'])
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    train_loader = DataLoader(dataset, batch_size=config['training']['batch_size'], 
                              sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # 2. 初始化 HMAE 任务处理器 (引擎)
    hmae_engine = HierarchicalMaskedAutoEncoder(
        visible_fraction=config['model']['visible_fraction'],
        fixed_length=config['model']['fixed_length'],
        patch_size=config['model']['patch_size'],
        norm_pix_loss=True
    )

    # 3. 初始化 HVIT 模型
    model = HVIT(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads']
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 4. 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'] * len(train_loader))
    
    # 5. 开始预训练
    pretrain_hmae_model(model, hmae_engine, train_loader, None, optimizer, scheduler, 
                        config['training']['num_epochs'], device, args, config, start_epoch=0, is_ddp=True)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMAE Pre-training Script")
    parser.add_argument('--config', type=str, default='./configs/hmae-vit-b_S8D.yaml', help='Path to YAML config')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hmae_s8d_pretrain_ddp(args, config)