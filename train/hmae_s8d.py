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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from matplotlib import pyplot as plt
import numpy as np
import cv2

# --- Project-specific Imports ---
sys.path.append("./")

from model import create_model
from gdt.hmae import HierarchicalMaskedAutoEncoder
from model.utilz import save_checkpoint, load_checkpoint
from dataset.s8d_hmae import S8DPretrainDataset


def setup_logging(config):
    """根据配置文件设置日志输出路径。"""
    base_dir = config['output']['base_dir']
    save_name = config['output']['save_name']
    log_dir = os.path.join(base_dir, save_name)
    os.makedirs(log_dir, exist_ok=True)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    # 清除旧的 handlers
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

def visualize_hmae_reconstruction(batch, pred_img, loss, step, output_dir, config, prefix="train"):
    """
    将 Patches 重新拼成完整图像，并展示 QDT 结构与 MAE 重建效果。
    实现了针对 S8D 灰度图的局部反归一化。
    """
    img_size = config['model']['img_size']
    norm_pix_loss = config['model'].get('norm_pix_loss', True)
    
    # 转换数据到 CPU 和 float32 供 Matplotlib 使用
    target_patches = batch['targets'][0].detach().cpu().float() # [L, C, P, P]
    coords = batch['coords'][0].detach().cpu().numpy().astype(int) # [L, 4]
    mask = batch['mask_vis'][0].detach().cpu().numpy()            # [L]
    recon_patches = pred_img[0].detach().cpu().float()            # [L, patch_dim]
    
    # 初始化对比画布
    canvas_target = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_input = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_recon = np.zeros((img_size, img_size), dtype=np.float32)
    canvas_qdt = np.zeros((img_size, img_size), dtype=np.float32)

    L = target_patches.shape[0]
    for i in range(L):
        x1, x2, y1, y2 = coords[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue

        # 1. 准备 Target 补丁
        t_patch_np = target_patches[i].permute(1, 2, 0).numpy().squeeze()
        resized_t = cv2.resize(t_patch_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
        canvas_target[y1:y2, x1:x2] = resized_t
        canvas_qdt[y1:y2, x1:x2] = resized_t
        cv2.rectangle(canvas_qdt, (x1, y1), (x2, y2), (1.0,), 1)

        # 2. 准备 Input (仅显示可见部分)
        if mask[i] == 0:
            canvas_input[y1:y2, x1:x2] = resized_t
        
        # 3. 准备 Reconstruction
        if mask[i] == 0:
            canvas_recon[y1:y2, x1:x2] = resized_t
        else:
            r_patch_reshaped = recon_patches[i].view_as(target_patches[i])
            r_patch_np = r_patch_reshaped.permute(1, 2, 0).numpy().squeeze()
            
            if norm_pix_loss:
                mean, std = t_patch_np.mean(), t_patch_np.std()
                r_patch_np = r_patch_np * (std + 1e-6) + mean
            
            r_patch_np = np.clip(r_patch_np, 0, 1)
            canvas_recon[y1:y2, x1:x2] = cv2.resize(r_patch_np, (w, h), interpolation=cv2.INTER_NEAREST)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Step: {step} | Loss: {loss:.6f}", fontsize=14)
    
    titles = ["Original", "QDT Grid", "MAE Input", "MAE Recon"]
    canvases = [canvas_target, canvas_qdt, canvas_input, canvas_recon]
    
    for ax, canvas, title in zip(axes, canvases, titles):
        ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}_full.png"))
    plt.close(fig)

def pretrain_hmae_model(model, hmae_engine, train_loader, optimizer, scheduler, device_id, config, is_ddp=False):
    """核心训练循环。"""
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    num_epochs = config['training']['num_epochs']
    is_main_process = not is_ddp or (dist.get_rank() == 0)
    output_path = os.path.join(config['output']['base_dir'], config['output']['save_name'])

    start_epoch = 0
    resume_path = os.path.join(output_path, "latest_checkpoint.pth")
    if os.path.exists(resume_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, scaler, resume_path, device_id)

    if is_main_process:
        logging.info(f"开始 HMAE 预训练模式：{config['model']['model_type']}")

    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device_id, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
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
            
            if (i + 1) % 200 == 0 and is_main_process:
                batch['mask_vis'] = mask 
                visualize_hmae_reconstruction(
                    batch, pred_img, loss.item() * accumulation_steps, i + 1,
                    os.path.join(output_path, "images"), config, prefix=f"train_e{epoch + 1}_b{i + 1}"
                )

            if (i + 1) % 10 == 0 and is_main_process:
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {(running_loss/10):.5f}')
                running_loss = 0.0

        if is_main_process:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, config, resume_path)

def hmae_s8d_pretrain_ddp(config):
    """主启动逻辑。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0: setup_logging(config)

    # 1. 实例化数据集
    dataset = S8DPretrainDataset(
        root_dir=config['dataset']['data_dir'], 
        img_size=config['model']['img_size'],
        hmae_config={
            'fixed_length': config['model']['fixed_length'],
            'patch_size': config['model']['patch_size']
        }
    )
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    train_loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        sampler=sampler, 
        num_workers=config['dataset']['num_workers'], 
        pin_memory=True
    )

    # 2. --- 使用工厂函数动态实例化模型 ---
    model_type = config['model'].get('model_type', 'hvit_b')
    model = create_model(
        model_type=model_type,
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_channels=config['model'].get('in_channels', 1),
        mask_ratio=config['model'].get('mask_ratio', 0.75),
        norm_pix_loss=config['model'].get('norm_pix_loss', True)
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        betas=tuple(config['training'].get('betas', [0.9, 0.95]))
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'] * len(train_loader))
    
    # 4. 初始化 Loss 引擎 (仅需参数)
    hmae_engine = HierarchicalMaskedAutoEncoder(
        fixed_length=config['model']['fixed_length'],
        patch_size=config['model']['patch_size']
    )

    # 5. 开始预训练
    pretrain_hmae_model(model, hmae_engine, train_loader, optimizer, scheduler, device, config, is_ddp=(world_size > 1))
    
    if world_size > 1: dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMAE Pre-training Script")
    parser.add_argument('--config', type=str, default='./configs/hmae-vit-b_S8D.yaml', help='Path to YAML config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hmae_s8d_pretrain_ddp(config)