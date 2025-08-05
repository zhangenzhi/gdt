import os
import sys
import yaml
import logging
import argparse
import contextlib
from typing import Dict

import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Transformer Engine and Apex Imports ---
# These are the key libraries for FP8 and optimized performance
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from apex.optimizers import FusedAdam

# Import the dataset functions
sys.path.append("./")
from dataset.imagenet import imagenet_distribute
from model.vit import create_vit_te_model

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

def train_vit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_id, args, config, start_epoch, best_val_acc=0.0, is_ddp=False):
    # GradScaler is NOT needed for Transformer Engine's FP8 training.
    # TE manages scaling internally via its recipe.
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info("Starting ViT training for %d epochs with Transformer Engine (FP8)...", num_epochs)
        logging.info(f"torch.compile: {config['training'].get('use_compile', False)}, Fused Optimizer: {config['training'].get('use_fused_optimizer', False)}")
        logging.info(f"Will start training from Epoch {start_epoch + 1}...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            sync_context = model.no_sync() if (is_ddp and is_accumulation_step) else contextlib.nullcontext()
            
            with sync_context:
                # The `te.fp8_autocast` context is now handled inside the model's forward pass.
                # We just call the model directly.
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps
                
                # No GradScaler needed. Just call backward() on the loss.
                loss.backward()

            if not is_accumulation_step:
                # No GradScaler needed. Just call step() and zero_grad().
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # Scheduler step is per optimizer step
                if scheduler: scheduler.step()

            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss.detach().item() * accumulation_steps # De-normalize for logging

            if (i + 1) % 10 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, current_lr: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0
                
        val_acc = evaluate_model_compatible(model, val_loader, device_id, is_ddp=is_ddp)
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | VAL ACC: {val_acc:.4f} | current_lr: {current_lr:.6f}")
            
            checkpoint_dir = os.path.join(args.output, args.savefile)
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, latest_checkpoint_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving best model...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Training. Best Validation Accuracy: {best_val_acc:.4f}')

def evaluate_model_compatible(model, val_loader, device, is_ddp=False):
    """Evaluation function compatible with TE models."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # No explicit autocast needed; the TE model handles its own precision context.
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

def setup_ddp():
    """Sets up DDP based on torchrun or SLURM environment variables."""
    if 'SLURM_PROCID' in os.environ: # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    else: # torchrun environment
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return local_rank, rank, world_size

def vit_imagenet_train_ddp(args, config):
    """Main DDP training function for the Transformer Engine ViT."""
    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    if rank == 0:
        setup_logging(args)
        logging.info(f"Starting DDP training with {world_size} processes (NCCL backend).")

    dataloaders = imagenet_distribute(
        img_size=config['model']['img_size'],
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )
    
    # Use the factory for the Transformer Engine model
    model = create_vit_te_model(config).to(device)
        
    if config['training'].get('use_compile', False):
        if rank == 0: logging.info("Applying torch.compile()...")
        # Compile after moving to device, before wrapping with DDP
        model = torch.compile(model)
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss(label_smoothing=config['training'].get('label_smoothing', 0.1))

    training_config = config['training']
    use_fused = training_config.get('use_fused_optimizer', False)
    if use_fused:
        if rank == 0: logging.info("Using Apex FusedAdam optimizer.")
        optimizer = FusedAdam(
            model.parameters(), 
            lr=training_config['learning_rate'], 
            weight_decay=training_config['weight_decay'],
            betas=tuple(training_config.get('betas', (0.9, 0.999)))
        )
    else:
        if rank == 0: logging.info("Using standard torch.optim.AdamW optimizer.")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=training_config['learning_rate'], 
            weight_decay=training_config['weight_decay'],
            betas=tuple(training_config.get('betas', (0.9, 0.999)))
        )
    
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 0)
    steps_per_epoch = len(dataloaders['train'])
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = warmup_epochs * steps_per_epoch
    
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_dir = os.path.join(args.output, args.savefile)
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if args.reload and os.path.exists(checkpoint_path):
        if rank == 0: logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if rank == 0: logging.info(f"Successfully resumed. Starting from Epoch {start_epoch + 1}.")
            
    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, num_epochs, device, args, config, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT FP8 Training Script with Transformer Engine")
    
    parser.add_argument('--config', type=str, default='./configs/vit_fp8.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='vit-b-16-fp8', help='Subdirectory for saving logs and models')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the latest checkpoint if it exists')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    # This script is intended for DDP training, launched with torchrun or srun
    vit_imagenet_train_ddp(args, config)