import os
import sys
import yaml
import logging
import argparse
import contextlib

import torch
from torch import nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast 

# --- Project-specific Imports ---
sys.path.append("./")
# Import the new SHF dataloader and model
from dataset.imagenet import build_shf_imagenet_dataloader
from model.vit import SHFVisionTransformer
# Utility for Layer-wise Rate Decay
from dataset.utlis import param_groups_lrd
from dataset.utlis import SHFMixup

def setup_logging(args):
    """Configures logging to file and console."""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "out.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def train_shf_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device_id, args, start_epoch, best_val_acc=0.0, is_ddp=False, mixup_fn=None):
    scaler = GradScaler(enabled=True)
    num_epochs = config['training']['num_epochs']
    
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info(f"Starting SHF-ViT training for {num_epochs} epochs...")
        logging.info(f"Will start training from Epoch {start_epoch + 1}...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (batch_dict, labels) in enumerate(train_loader):
            # Move all tensors in the dictionary to the device
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            # --- [KEY CHANGE] Apply Mixup/CutMix ---
            # Store original labels for accuracy calculation
            original_labels = labels.clone()
            if mixup_fn is not None:
                batch_dict, soft_labels = mixup_fn(batch_dict, labels)
            else:
                # If not using mixup, convert to one-hot for consistency with LabelSmoothingCrossEntropy
                soft_labels = labels

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(batch_dict)
                # Use the (potentially soft) labels for loss calculation
                loss = criterion(outputs, soft_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler: scheduler.step()
                
            _, predicted = torch.max(outputs.data, 1)
            # Use original labels for calculating accuracy
            running_total += original_labels.size(0)
            running_corrects += (predicted == original_labels).sum().item()
            running_loss += loss.item()

            if (i + 1) % 50 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 50
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, LR: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        val_acc = evaluate_shf_model(model, val_loader, device_id, is_ddp=is_ddp)
                
        if is_main_process:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f}")
            
            checkpoint_dir = os.path.join(args.output, args.savefile)
            checkpoint_data = {
                'epoch': epoch, 'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save(checkpoint_data, latest_checkpoint_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving best model...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(checkpoint_data, best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Training. Best Validation Accuracy: {best_val_acc:.4f}')


def evaluate_shf_model(model, val_loader, device, is_ddp=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch_dict, labels in val_loader:
                for key, value in batch_dict.items():
                    if isinstance(value, torch.Tensor):
                        batch_dict[key] = value.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(batch_dict)
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def shf_imagenet_train(args, config):
    """Main DDP training function for SHF-ViT on a SLURM-based supercomputer."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup_ddp(rank, world_size)
    
    local_rank = int(os.environ['SLURM_LOCALID'])
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    if rank == 0: setup_logging(args)
    logging.info(f"DDP Initialized for SHF-ViT. Rank {rank}/{world_size} on device {device_id}")

    dataloaders = build_shf_imagenet_dataloader(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        fixed_length=config['model']['fixed_length'],
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )
    
    model = SHFVisionTransformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        drop_path_rate = config['model']['drop_path_rate']
    ).to(device_id)
        
    model = DDP(model, device_ids=[device_id])

    mixup_fn = None
    # --- [KEY CHANGE] Setup Mixup and Loss Function ---
    mixup_fn = None
    if config['training'].get('use_mixup', False):
        mixup_fn = SHFMixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=0.1,
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size']
        )
        # Use SoftTargetCrossEntropy for soft labels created by Mixup/CutMix
        criterion = SoftTargetCrossEntropy()
    else:
        # Use standard LabelSmoothingCrossEntropy if Mixup is disabled
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
    model_without_ddp = model.module
    no_weight_decay_list = hasattr(model_without_ddp, 'no_weight_decay') and model_without_ddp.no_weight_decay() or set()
    param_groups = param_groups_lrd(model_without_ddp, config['training']['weight_decay'],
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=0.75
    )
    use_fused = config['training'].get('use_fused_optimizer', False)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['training']['learning_rate'], 
        betas=tuple(config['training'].get('betas', (0.9, 0.95))),
        fused=use_fused
    )

    training_config = config['training']
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
        scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
    
    start_epoch = 0
    best_val_acc = 0.0
    checkpoint_dir = os.path.join(args.output, args.savefile)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if args.reload and os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if dist.get_rank() == 0:
            logging.info(f"Successfully resumed. Starting from Epoch {start_epoch + 1}. Best Acc: {best_val_acc:.4f}")
            
    train_shf_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device_id, args, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=True, mixup_fn=mixup_fn)
    dist.destroy_process_group()
    
def shf_imagenet_train_single(args, config):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"Starting SHF-ViT training with {world_size} GPUs.")

    dataloaders = build_shf_imagenet_dataloader(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        fixed_length=config['model']['fixed_length'],
        data_dir=args.data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )
    
    model = SHFVisionTransformer(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads']
    ).to(device)
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # --- [KEY CHANGE] Setup Mixup and Loss Function ---
    mixup_fn = None
    if config['training'].get('use_mixup', False):
        mixup_fn = SHFMixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            switch_prob=0.5,
            label_smoothing=0.1,
            num_classes=config['model']['num_classes'],
            img_size=config['model']['img_size']
        )
        # Use SoftTargetCrossEntropy for soft labels created by Mixup/CutMix
        # criterion = SoftTargetCrossEntropy()
        criterion = nn.CrossEntropyLoss()
    else:
        # Use standard LabelSmoothingCrossEntropy if Mixup is disabled
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # --- [NEW] Advanced Optimizer Setup ---
    model_without_ddp = model.module
    # Assuming the SHF model has a `no_weight_decay` method similar to timm models
    no_weight_decay_list = hasattr(model_without_ddp, 'no_weight_decay') and model_without_ddp.no_weight_decay() or set()
    param_groups = param_groups_lrd(model_without_ddp, config['training']['weight_decay'],
        no_weight_decay_list=no_weight_decay_list,
        layer_decay=0.65
    )
    use_fused = config['training'].get('use_fused_optimizer', False)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['training']['learning_rate'], 
        betas=tuple(config['training'].get('betas', (0.9, 0.95))),
        fused=use_fused
    )

    # --- [NEW] Advanced Scheduler Setup ---
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
    best_val_acc = 0.0
    checkpoint_dir = os.path.join(args.output, args.savefile)
    # Load from the best model checkpoint to ensure we resume from the best state
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if args.reload and os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0) # Use .get for backward compatibility
        
        if dist.get_rank() == 0:
            logging.info(f"Successfully resumed. Starting from Epoch {start_epoch + 1}. Best Acc: {best_val_acc:.4f}")
            
    train_shf_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=True, mixup_fn=mixup_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHF-ViT Training Script")
    parser.add_argument('--config', type=str, default='./configs/shf-vit-b_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='shf-vit-b16-4k-pz8', help='Subdirectory for saving logs and models')
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    args.output = os.path.join(args.output, config.get('task_name', 'shf_train'))
    os.makedirs(args.output, exist_ok=True)
    
    shf_imagenet_train_single(args, config)
