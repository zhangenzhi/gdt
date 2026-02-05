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

# --- Project-specific Imports ---
sys.path.append("./")

from model.hvit import HMAEVIT as HVIT
from model.utilz import save_checkpoint, load_checkpoint
from dataset.s8d_hmae import S8DPretrainDataset


def setup_logging(config):
    """Configures logging using paths from the config file."""
    base_dir = config['output']['base_dir']
    save_name = config['output']['save_name']
    log_dir = os.path.join(base_dir, save_name)
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

def visualize_hmae_reconstruction(processed_batch, pred_img, loss, step, output_dir, config, prefix="train"):
    """
    将补丁重新拼成原图，并进行局部的反归一化处理以保证可视化效果。
    """
    img_size = config['model']['img_size']
    norm_pix_loss = config['model'].get('norm_pix_loss', True)
    
    # 转换为 float32 避免绘图报错
    target_patches = processed_batch['targets'][0].detach().cpu().float() # [L, C, P, P]
    coords = processed_batch['coords'][0].detach().cpu().numpy().astype(int)
    mask = processed_batch['mask_vis'][0].detach().cpu().numpy()
    recon_patches = pred_img[0].detach().cpu().float() # [L, patch_dim]
    
    canvas_target = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_input = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_recon = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_qdt = np.zeros((img_size, img_size), dtype=np.float32)

    L = target_patches.shape[0]
    for i in range(L):
        m_val = mask[i]
        x1, x2, y1, y2 = coords[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue

        # 1. 原始补丁 (Target)
        t_patch_np = target_patches[i].permute(1, 2, 0).numpy().squeeze()
        resized_t = cv2.resize(t_patch_np, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas_target[y1:y2, x1:x2] = resized_t
        
        # 2. QDT 网格结构
        canvas_qdt[y1:y2, x1:x2] = resized_t
        cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), (1.0,), 1)

        # 3. 输入补丁 (Input - MAE 风格仅显示可见部分)
        if m_val == 0:
            canvas_input[y1:y2, x1:x2] = resized_t
        
        # 4. 重建补丁 (Reconstruction)
        if m_val == 0:
            canvas_recon[y1:y2, x1:x2] = resized_t
        else:
            # 这里的核心逻辑：反归一化
            r_patch_reshaped = recon_patches[i].view_as(target_patches[i])
            r_patch_np = r_patch_reshaped.permute(1, 2, 0).numpy().squeeze()
            
            if norm_pix_loss:
                # 使用目标补丁的局部统计量进行反归一化
                mean = t_patch_np.mean()
                std = t_patch_np.std()
                r_patch_np = r_patch_np * (std + 1e-6) + mean
            
            r_patch_np = np.clip(r_patch_np, 0, 1)
            canvas_recon[y1:y2, x1:x2] = cv2.resize(r_patch_np, (w, h), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(canvas_target, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Original (Target)")
    
    axes[1].imshow(canvas_qdt, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("QDT Structure")
    
    axes[2].imshow(canvas_input, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("HMAE Input (Visible)")
    
    axes[3].imshow(canvas_recon, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Reconstructed")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}_full.png"))
    plt.close(fig)

def pretrain_hmae_model(model, train_loader, optimizer, scheduler, device_id, config, start_epoch=0, is_ddp=False):
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    num_epochs = config['training']['num_epochs']
    is_main_process = not is_ddp or (dist.get_rank() == 0)
    
    output_path = os.path.join(config['output']['base_dir'], config['output']['save_name'])

    # 如果有 checkpoint 则恢复
    resume_path = os.path.join(output_path, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, resume_path, device_id)

    if is_main_process:
        logging.info(f"Starting HMAE pre-training for {num_epochs} epochs (MAE Style)...")
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # 将数据移动到 GPU
            batch = {k: v.to(device_id, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # 1. 直接前向传播
                    # 模型内部现在处理了 masking 和 loss 计算
                    loss, pred_img, mask = model(
                        batch['patches'], batch['coords'], batch['depths']
                    )
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            if (i + 1) % 500 == 0 and is_main_process:
                # 为了可视化，我们将模型生成的 mask 存入 batch
                batch['mask_vis'] = mask
                visualize_hmae_reconstruction(
                    batch, pred_img, loss.item() * accumulation_steps, i + 1,
                    os.path.join(output_path, "images"), config, prefix=f"train_e{epoch + 1}"
                )

            if (i + 1) % 10 == 0 and is_main_process:
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {(running_loss/10):.5f}')
                running_loss = 0.0

        if is_main_process:
            checkpoint_path = os.path.join(output_path, "latest_checkpoint.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, checkpoint_path)

def hmae_s8d_pretrain_ddp(config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0: setup_logging(config)

    # 1. 加载数据
    dataset = S8DPretrainDataset(
        root_dir=config['dataset']['data_dir'], 
        img_size=config['model']['img_size']
    )
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    train_loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler, 
        num_workers=config['dataset']['num_workers'], 
        pin_memory=True
    )

    # 2. 初始化重构后的模型
    model = HVIT(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_channels=config['model'].get('in_channels', 1),
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads'],
        mask_ratio=config['model'].get('mask_ratio', 0.75)
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'] * len(train_loader))
    
    # 3. 运行训练
    pretrain_hmae_model(model, train_loader, optimizer, scheduler, device, config, is_ddp=(world_size > 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMAE Pre-training Script")
    parser.add_argument('--config', type=str, default='./configs/hmae-vit-b_S8D.yaml', help='Path to YAML config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hmae_s8d_pretrain_ddp(config)