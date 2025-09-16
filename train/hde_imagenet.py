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
from matplotlib import pyplot as plt
import numpy as np
import cv2

# --- Project-specific Imports ---
# Assume these modules are in the python path
sys.path.append("./")
# NOTE: You will need to create/provide your HDE model implementation here
# from model.hde import HDE 

from model.hde import HDEVIT as HDE
from dataset.imagenet import build_hde_imagenet_dataloaders
from gdt.hde import Rect # Needed for visualization

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

def visualize_hde_reconstruction(batch, reconstructed_patches, loss, step, output_dir, prefix="train"):
    """
    Creates and saves a visualization for the HDE task.
    Shows Original, HDE Input, and Reconstructed Image.
    """
    # --- 1. Prepare Data from Batch (visualize first image) ---
    original_img_tensor = batch['original_image'][0].cpu()
    patches_tensor = batch['patches'][0].cpu()
    coords_tensor = batch['coords'][0].cpu()
    mask_tensor = batch['mask'][0].cpu()
    recon_patches_tensor = reconstructed_patches[0].cpu()

    img_size = original_img_tensor.shape[1]
    
    # --- 2. Denormalize Tensors for Plotting ---
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def denorm(tensor):
        t = tensor.permute(1, 2, 0).numpy()
        t = std * t + mean
        return np.clip(t, 0, 1)

    original_img_np = denorm(original_img_tensor)

    # --- 3. Create HDE Input Visualization Canvas ---
    hde_input_canvas = np.zeros_like(original_img_np)
    for i in range(coords_tensor.shape[0]):
        x1, x2, y1, y2 = coords_tensor[i].numpy()
        if (x2 - x1 == 0) or (y2 - y1 == 0): continue

        is_noised = mask_tensor[i].item() == 1
        
        if is_noised:
            # For noised patches, we show the input patch from the dataloader
            patch_np = denorm(patches_tensor[i])
        else:
            # For visible patches, we show the original content
            patch_np = original_img_np[y1:y2, x1:x2, :]

        # Resize patch to its original quadtree size before placing on canvas
        resized_patch = cv2.resize(patch_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        hde_input_canvas[y1:y2, x1:x2, :] = resized_patch

    # --- 4. Create Reconstructed Image Canvas ---
    reconstructed_img = original_img_np.copy()
    num_noised = recon_patches_tensor.shape[0]
    
    noised_coords = coords_tensor[mask_tensor == 1]
    
    for i in range(num_noised):
        x1, x2, y1, y2 = noised_coords[i].numpy()
        if (x2 - x1 == 0) or (y2 - y1 == 0): continue
        
        # Denormalize the single reconstructed patch
        recon_patch_np = denorm(recon_patches_tensor[i])
        
        resized_recon_patch = cv2.resize(recon_patch_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        reconstructed_img[y1:y2, x1:x2, :] = resized_recon_patch

    # --- 5. Plot and Save ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Loss: {loss:.4f} | Step: {step}", fontsize=16)
    
    axes[0].imshow(original_img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(hde_input_canvas)
    axes[1].set_title("HDE Input (Clean & Noised)")
    axes[1].axis('off')

    axes[2].imshow(reconstructed_img)
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}.png"))
    plt.close(fig)


def pretrain_hde_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device_id, args, config, start_epoch, is_ddp=False):
    scaler = GradScaler(enabled=True)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = not is_ddp or (dist.get_rank() == 0)

    if is_main_process:
        logging.info(f"Starting HDE pre-training for {num_epochs} epochs with AMP...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Move all tensor data to the device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device_id, non_blocking=True)
            
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # HDE model forward pass takes the whole batch dictionary
                    loss, recon_patches, mask = model(batch)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            if (i + 1) % 500 == 0 and is_main_process:
                with torch.no_grad():
                    loss_val, recon, _ = model(batch)
                    visualize_hde_reconstruction(
                        batch, 
                        recon,
                        loss_val.item(), 
                        i + 1,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"train_e{epoch + 1}"
                    )

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.5f}, LR: {current_lr:.6f}')
                running_loss = 0.0

        val_loss = evaluate_hde_model(model, val_loader, device_id, args, is_ddp=is_ddp, epoch=epoch)
                
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
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict()}, best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Pre-training. Best Validation Loss: {best_val_loss:.5f}')


def evaluate_hde_model(model, val_loader, device, args, is_ddp=False, epoch=0):
    model.eval()
    total_loss, total_samples = 0, 0
    is_main_process = not is_ddp or (dist.get_rank() == 0)
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for i, batch in enumerate(val_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device, non_blocking=True)
                
                loss, recon, mask = model(batch)
                total_loss += loss.item() * batch['patches'].size(0)
                total_samples += batch['patches'].size(0)
                
                if i < 5 and is_main_process:
                     visualize_hde_reconstruction(
                        batch, 
                        recon,
                        loss.item(), 
                        i,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"val_e{epoch + 1}"
                    )

    if is_ddp:
        # Gather metrics from all processes
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(total_samples_tensor)
        total_loss, total_samples = total_loss_tensor.item(), total_samples_tensor.item()
        
    return total_loss / total_samples if total_samples > 0 else 0

def hde_imagenet_pretrain_ddp(args, config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"Starting DDP pre-training with {world_size} GPUs.")

    dataloaders = build_hde_imagenet_dataloaders(
        img_size=config['model']['img_size'],
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers,
        fixed_length= (config['model']['img_size'] // config['model']['patch_size']) ** 2
    )

    model = HDE(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads'],
        # Pass visible_fraction instead of mask_ratio
        visible_fraction=config['model']['visible_fraction']
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', (0.9, 0.95)))
    )
    
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=config['training'].get('eta_min', 1e-6))
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    start_epoch = 0
    # Add checkpoint loading logic if needed
            
    pretrain_hde_model(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, num_epochs, device, args, config, start_epoch=start_epoch, is_ddp=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDE Pre-training Script")
    parser.add_argument('--config', type=str, default='./configs/hde-vit-b_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='hde_vit-b16', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output = os.path.join(args.output, config.get('task_name', 'hde_pretrain'))
    os.makedirs(args.output, exist_ok=True)
    
    hde_imagenet_pretrain_ddp(args, config)
