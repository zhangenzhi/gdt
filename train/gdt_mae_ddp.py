import os
import sys
import yaml
import logging
import argparse
import math
from typing import List, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import transforms
from PIL import Image

# Ensure project root is in path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.imagenet import imagenet_distribute, imagenet_subloaders
from model.gdt_mae import create_gdt_mae, visualize_reconstruction

def setup_logging(args):
    """Configures logging to file and console."""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Avoid duplicate handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "pretrain.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_mae_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device_id, args, config, is_ddp=False):
    """Main training loop for the MAE pre-training task."""
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)
    
    if is_main_process:
        logging.info("Starting MAE pre-training for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        for i, (images, _) in enumerate(train_loader): # Labels are ignored in MAE
            images = images.to(device_id, non_blocking=True)
            
            optimizer.zero_grad()
            loss, _, _ = model(images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 50 == 0 and is_main_process:
                avg_loss = running_loss / 50
                logging.info(f'[Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}] Pre-train Loss: {avg_loss:.4f}')
                running_loss = 0.0
                
        scheduler.step()

        if is_main_process:
            logging.info(f"Epoch {epoch + 1} finished. LR: {optimizer.param_groups[0]['lr']:.6f}")
            # --- Periodically save model and visualization ---
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                # Save the pre-trained encoder's state dict
                encoder_path = os.path.join(args.output, args.savefile, f"pretrained_encoder_epoch_{epoch+1}.pth")
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.encoder.state_dict(), encoder_path)
                logging.info(f"Saved pre-trained encoder to {encoder_path}")

                # Generate visualization using one image from the validation set
                model.eval()
                with torch.no_grad():
                    val_images, _ = next(iter(val_loader))
                    val_image_tensor = val_images[0].unsqueeze(0).to(device_id)
                    
                    _, recon_patches, visible_indices = model(val_image_tensor)
                    
                    viz_filename = os.path.join(args.output, args.savefile, f"reconstruction_epoch_{epoch+1}.png")
                    
                    # For visualization, we need an un-normalized image
                    inv_normalize = transforms.Normalize(
                       mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
                       std=[1/0.5, 1/0.5, 1/0.5]
                    )
                    # Note: The original dataloader might have normalization.
                    # For accurate visualization, it's best to reload the image from disk.
                    # We will assume val_images are normalized for now.
                    img_for_viz = inv_normalize(val_images[0])


                    visualize_reconstruction(
                        img_tensor=img_for_viz,
                        recon_patches=recon_patches,
                        visible_indices=visible_indices,
                        initial_patch_size=model_to_save.encoder.initial_patch_size,
                        output_filename=viz_filename
                    )
                model.train() # Set back to training mode

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def gdt_mae_pretrain(args, config):
    """Main DDP pre-training function."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup_ddp(rank, world_size)
    
    local_rank = int(os.environ['SLURM_LOCALID'])
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    if rank == 0:
        setup_logging(args)
    logging.info(f"DDP Initialized for MAE Pre-training. Rank {rank}/{world_size} on device {device_id}")

    dataloaders = imagenet_distribute(args=args)
    model = create_gdt_mae(config)
    
    model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    train_mae_model(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, config['training']['num_epochs'], device_id, args, config, is_ddp=True)
    dist.destroy_process_group()
    
def gdt_mae_pretrain_local(args, config):
    """Main function for local (non-DDP) pre-training."""
    setup_logging(args)
    
    device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        logging.info(f"Starting local MAE pre-training on device cuda:{device_id}")
    else:
        device_id = "cpu"
        logging.info(f"Starting local MAE pre-training on device {device_id}")

    dataloaders, _ = imagenet_subloaders(subset_data_dir=args.data_dir, batch_size=config['training']['batch_size'], num_workers=args.num_workers)

    model = create_gdt_mae(config)
    model.to(device_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    train_mae_model(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, config['training']['num_epochs'], device_id, args, config, is_ddp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDT-MAE Pre-training Script")
    
    parser.add_argument('--config', type=str, default='./configs/gdt_mae_test.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='gdt_mae_test', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='gdt_mae_pretrain_test', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help='Path to the dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from a checkpoint (not implemented for pre-training yet).')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update args from config for consistency if needed
    # (e.g., batch_size can be overridden by command line)
    config['training']['batch_size'] = getattr(args, 'batch_size', config['training']['batch_size'])
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    # This script is designed to be launched by main.py, but this allows direct local testing.
    gdt_mae_pretrain_local(args, config)
