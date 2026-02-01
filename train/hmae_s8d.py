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
from gdt.hmae import HierarchicalMaskedAutoEncoder
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

def visualize_hmae_reconstruction(processed_batch, pred_img, loss, step, output_dir, prefix="train"):
    """
    Creates and saves a visualization for the HMAE task.
    Shows: Target Patches, Noised Input Patches, and Reconstructed Patches.
    """
    target_patches = processed_batch['targets'][0].cpu()
    input_patches = processed_batch['patches'][0].cpu()
    mask = processed_batch['mask'][0].cpu()
    recon_patches = pred_img[0].cpu()
    
    # Select up to 16 noised patches for visualization
    indices = torch.where(mask == 1)[0][:16]
    if len(indices) == 0: return

    num_plot = len(indices)
    fig, axes = plt.subplots(num_plot, 3, figsize=(9, 2 * num_plot))
    
    # Handle single patch case for subplot indexing
    if num_plot == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices):
        # Target (Clean)
        axes[i, 0].imshow(target_patches[idx].permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title("Target")
        # Input (Noised)
        axes[i, 1].imshow(input_patches[idx].permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title("Input")
        # Reconstruction
        axes[i, 2].imshow(recon_patches[idx].detach().permute(1, 2, 0).numpy().squeeze(), cmap='gray')
        axes[i, 2].set_title("Recon")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}.png"))
    plt.close(fig)

def run_pretrain_batch(batch, model, hmae_engine, device_id):
    """
    Helper to process raw images through the engine and model.
    """
    images = batch['image'] # [B, 1, H, W]
    edges = batch['edges']  # [B, 1, H, W]
    
    batch_patches, batch_targets, batch_noises, batch_coords, batch_depths, batch_masks = [], [], [], [], [], []
    
    # CPU-based processing via the HMAE Engine
    imgs_np = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
    edges_np = (edges.squeeze(1).cpu().numpy() * 255).astype('uint8')

    for b in range(images.shape[0]):
        p, t, n, c, d, m = hmae_engine.process_single(imgs_np[b], edges_np[b])
        batch_patches.append(p)
        batch_targets.append(t)
        batch_noises.append(n)
        batch_coords.append(c)
        batch_depths.append(d)
        batch_masks.append(m)

    processed = {
        'patches': torch.stack(batch_patches).to(device_id, non_blocking=True),
        'targets': torch.stack(batch_targets).to(device_id, non_blocking=True),
        'noises': torch.stack(batch_noises).to(device_id, non_blocking=True),
        'coords': torch.stack(batch_coords).to(device_id, non_blocking=True),
        'depths': torch.stack(batch_depths).to(device_id, non_blocking=True),
        'mask': torch.stack(batch_masks).to(device_id, non_blocking=True)
    }

    pred_img, pred_noise = model(processed['patches'], processed['coords'], processed['depths'], processed['mask'])
    loss = hmae_engine.train_step_loss(processed['targets'], pred_img, processed['noises'], pred_noise, processed['mask'])
    
    return loss, pred_img, processed

def pretrain_hmae_model(model, hmae_engine, train_loader, val_loader, optimizer, scheduler, device_id, config, start_epoch=0, is_ddp=False):
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    num_epochs = config['training']['num_epochs']
    is_main_process = not is_ddp or (dist.get_rank() == 0)
    
    output_path = os.path.join(config['output']['base_dir'], config['output']['save_name'])

    if is_main_process:
        logging.info(f"Starting HMAE pre-training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss, pred_img, processed = run_pretrain_batch(batch, model, hmae_engine, device_id)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            if (i + 1) % 500 == 0 and is_main_process:
                visualize_hmae_reconstruction(
                    processed, pred_img, loss.item() * accumulation_steps, i + 1,
                    os.path.join(output_path, "images"),
                    prefix=f"train_e{epoch + 1}"
                )

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.5f}, LR: {current_lr:.6f}')
                running_loss = 0.0

        if is_main_process:
            checkpoint_path = os.path.join(output_path, "latest_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)

def hmae_s8d_pretrain_ddp(config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(config)
        logging.info(f"HMAE Pre-training initialized. World Size: {world_size}")

    # 1. Dataset & DataLoader (Parameters from Config)
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
        pin_memory=True,
        shuffle=(sampler is None)
    )

    # 2. HMAE Task Engine
    hmae_engine = HierarchicalMaskedAutoEncoder(
        visible_fraction=config['model']['visible_fraction'],
        fixed_length=config['model']['fixed_length'],
        patch_size=config['model']['patch_size'],
        norm_pix_loss=True
    )

    # 3. Model
    model = HVIT(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        in_channels=config['model'].get('in_channels', 1),
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads']
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']), 
        weight_decay=float(config['training']['weight_decay']),
        betas=tuple(config['training'].get('betas', [0.9, 0.95]))
    )
    
    total_steps = config['training']['num_epochs'] * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(config['training'].get('eta_min', 1e-6)))
    
    # 5. Execute Training
    pretrain_hmae_model(
        model=model, 
        hmae_engine=hmae_engine, 
        train_loader=train_loader, 
        val_loader=None, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device_id=device, 
        config=config, 
        is_ddp=(world_size > 1)
    )
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMAE Pre-training Script")
    parser.add_argument('--config', type=str, default='./configs/hmae-vit-b-s8d.yaml', help='Path to YAML config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    hmae_s8d_pretrain_ddp(config)