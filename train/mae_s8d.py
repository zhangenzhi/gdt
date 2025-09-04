import os
import sys
import yaml
import logging
import argparse
import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("./")

from model.timm_mae import MAE
from dataset.s8d import build_s8d_dataloaders
from dataset.utlis import param_groups_lrd

def setup_logging(args):
    """Configures logging to file and console."""
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')


# <--- MODIFIED: This function is completely rewritten for single-channel grayscale images
def visualize_and_save(original_img, mask, recon_patches, patch_size, loss, step, output_dir, prefix="train"):
    """
    Creates and saves a visualization for single-channel grayscale images.
    """
    # --- 1. Denormalize and prepare tensors ---
    # Denormalize using mean=0.5, std=0.5
    original_img = original_img.cpu().to(torch.float32) * 0.5 + 0.5
    original_img = torch.clip(original_img, 0, 1)

    # Convert to NumPy array (H, W) for plotting
    original_img_hw = original_img.squeeze(0).numpy()

    recon_patches = recon_patches.cpu().to(torch.float32).numpy()
    mask = mask.cpu().numpy()

    H, W = original_img_hw.shape
    N = mask.shape[0] # Total number of patches
    num_patches_w = W // patch_size

    # --- 2. Create masked image ---
    masked_img = original_img_hw.copy()
    for i in range(N):
        if mask[i]: # If the patch is masked
            h_idx = i // num_patches_w
            w_idx = i % num_patches_w
            start_h, start_w = h_idx * patch_size, w_idx * patch_size
            masked_img[start_h : start_h + patch_size, start_w : start_w + patch_size] = 0.2 # Use a dark gray for masked areas

    # --- 3. Create reconstructed image ---
    reconstructed_img = original_img_hw.copy()
    
    # Reshape reconstructed patches. Shape: (N, patch_size, patch_size)
    recon_patches_reshaped = recon_patches.reshape(N, patch_size, patch_size)

    for i in range(N):
        if mask[i]: # If the patch was masked, fill it with the reconstruction
            h_idx = i // num_patches_w
            w_idx = i % num_patches_w
            start_h, start_w = h_idx * patch_size, w_idx * patch_size

            # Denormalize the reconstructed patch
            original_patch = original_img_hw[start_h : start_h + patch_size, start_w : start_w + patch_size]
            mean, std = original_patch.mean(), original_patch.std()
            denorm_patch = recon_patches_reshaped[i] * (std + 1e-6) + mean

            reconstructed_img[start_h : start_h + patch_size, start_w : start_w + patch_size] = np.clip(denorm_patch, 0, 1)

    # --- 4. Plot and save ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Loss: {loss:.4f} | Step: {step}", fontsize=16)

    axes[0].imshow(original_img_hw, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(masked_img, cmap='gray')
    axes[1].set_title("Masked Input")
    axes[1].axis('off')

    axes[2].imshow(reconstructed_img, cmap='gray')
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}.png"))
    plt.close(fig)


def pretrain_mae_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device_id, args, config, start_epoch, is_ddp=False):
    """
    MAE pre-training loop.
    """
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info("Starting MAE pre-training for %d epochs with AMP...", num_epochs)
        logging.info(f"Effective batch size: {args.batch_size * accumulation_steps * dist.get_world_size()}")
        logging.info(f"Will start training from Epoch {start_epoch + 1}...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        
        # <--- MODIFIED: Our dataloader only returns images, no labels
        for i, images in enumerate(train_loader):
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    images = images.to(device_id, non_blocking=True)
                    loss, recon_patches_flat, mask = model(images)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step() # Modified for simpler scheduler update
            
            running_loss += loss.item() * accumulation_steps
            
            if (i + 1) % 500 == 0 and is_main_process:
                with torch.no_grad():
                    loss_val, recon, mask_val = model(images)
                    visualize_and_save(
                        images[0], 
                        mask_val[0], 
                        recon[0], 
                        args.patch_size, 
                        loss_val.item(), 
                        i + 1,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"train_e{epoch + 1}"
                    )

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.5f}, current_lr: {current_lr:.6f}')
                running_loss = 0.0

        val_loss = evaluate_mae_model(model, val_loader, device_id, args, is_ddp=is_ddp, epoch=epoch)
                
        if is_main_process:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.5f}")
            
            checkpoint_dir = os.path.join(args.output, args.savefile)
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, latest_checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"New best validation loss: {best_val_loss:.5f}. Saving best model...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.module.state_dict(), best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Pre-training. Best Validation Loss: {best_val_loss:.5f}')


def evaluate_mae_model(model, val_loader, device, args, is_ddp=False, epoch=0):
    model.eval()
    total_loss, total_samples = 0, 0
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # <--- MODIFIED: Our dataloader only returns images, no labels
            for i, images in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                loss, recon, mask = model(images)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                if i < 10 and is_main_process:
                     visualize_and_save(
                        images[0], mask[0], recon[0], 
                        args.patch_size, loss.item(), i,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"val_e{epoch + 1}"
                    )

    if is_ddp:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(total_samples_tensor)
        total_loss, total_samples = total_loss_tensor.item(), total_samples_tensor.item()
        
    return total_loss / total_samples if total_samples > 0 else 0


def mae_pretrain_ddp(args, config): # <--- MODIFIED: Renamed function
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"Starting pre-training with {world_size} GPUs.")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    args.patch_size = config['model']['patch_size']
    
    # <--- MODIFIED: Call our new dataloader builder
    dataloaders = build_s8d_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.img_size
    )

    # <--- MODIFIED: Explicitly pass in_chans to the model constructor
    model = MAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=config['model']['in_channels'], # Ensure model uses 1 channel
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads'],
        mask_ratio=config['model']['mask_ratio']
    ).to(device)

    if config['training'].get('use_compile', False):
        if dist.get_rank() == 0: logging.info("Applying torch.compile()...")
        model = torch.compile(model)
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    param_groups = param_groups_lrd(model.module, config['training']['weight_decay'], layer_decay=0.75)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['training']['learning_rate'], 
        betas=tuple(config['training'].get('betas', (0.9, 0.99)))
    )
    
    training_config = config['training']
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 0)
    steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    start_epoch = 0
    if args.reload:
        # Simplified reload logic, adjust if needed
        checkpoint_path = os.path.join(args.output, args.savefile, "latest_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            if dist.get_rank() == 0: logging.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
    pretrain_mae_model(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, num_epochs, device, args, config, start_epoch=start_epoch, is_ddp=(world_size > 1))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE S8D Pre-training Script")
    
    parser.add_argument('--config', type=str, default='./configs/mae-vit-b128_S8D.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='mae_s8d_pretrain', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='mae_vit-p128-s8d', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/s8d/pretrain", help='Path to the S8D dataset directory')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the latest checkpoint')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output = os.path.join(args.output, args.task, args.savefile)
    os.makedirs(args.output, exist_ok=True)
    
    # <--- MODIFIED: Call the renamed main function
    mae_pretrain_ddp(args, config)
