import os
import sys
import yaml # For loading the config file
import logging
import argparse

import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

# Assume your model and dataset files are in these locations
sys.path.append("./")
from dataset.imagenet import imagenet_distribute, imagenet_subloaders
from model.gdt_vit import create_gdt_cls, visualize_evaluation_focus

def setup_logging(args):
    """Configures logging to file and console."""
    log_dir = os.path.join(args.output, args.savefile)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - RANK {} - %(levelname)s - %(message)s'.format(dist.get_rank() if dist.is_initialized() else 0),
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "out.log")),
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_id, args, config, is_ddp=False):
    """Main training loop with improved logging, checkpointing, and metrics."""
    model.train()
    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    # Only the main process should log general info
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)
    
    logging.info("Starting training for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        # --- FIXED: Conditionally set epoch for the sampler ---
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        
        # import pdb; pdb.set_trace()
        for i, (images, labels) in enumerate(train_loader):
            model.train()
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(images) # Assuming model returns (logits, viz_data)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # --- Calculate Training Accuracy ---
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i+1) % 10 == 0 and device_id == 0:
                train_acc = 100 * running_corrects / running_total
                avg_loss = running_loss / 10
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%')
                running_loss = 0.0
                running_corrects = 0
                running_total = 0
                
        # Update learning rate
        scheduler.step()

        # --- Validate after each epoch ---
        val_acc = evaluate_model(model, val_loader, device_id, args, config, is_ddp)
        
        if is_main_process:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # --- FIXED: Conditionally save the model ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving model to {checkpoint_path}")
                # Unwrap the model before saving if it's a DDP model
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(), checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Training. Best Validation Accuracy: {best_val_acc:.4f}')

def evaluate_model(model, val_loader, device_id, args, config, is_ddp=False):
    """Evaluates the model, correctly handling DDP synchronization if needed."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            outputs, leaf_nodes_data = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # log_dir = os.path.join(args.output, args.savefile)
    # visualize_evaluation_focus(
    #     original_image=images[0],
    #     leaf_nodes_data=leaf_nodes_data[0],
    #     target_leaf_size=config['classifier']['target_leaf_size'],
    #     img_size=args.img_size,
    #     output_filename=log_dir
    # )
        
    # --- FIXED: Conditionally synchronize results across all GPUs in DDP ---
    if is_ddp:
        total_tensor = torch.tensor(total, device=device_id)
        correct_tensor = torch.tensor(correct, device=device_id)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total = total_tensor.item()
        correct = correct_tensor.item()

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def setup_ddp(rank, world_size):
    """Initializes the distributed process group."""
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], os.environ['WORLD_SIZE'], os.environ['RANK'], local_rank))
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=int(os.environ['RANK']))
    print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))
    
def gdt_imagenet_train(args, config):
    """Main DDP training function."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup_ddp(rank, world_size)
    
    local_rank = int(os.environ['SLURM_LOCALID'])
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    if rank == 0:
        setup_logging(args)
    
    logging.info(f"DDP Initialized. Rank {rank}/{world_size} on device {device_id}")

    dataloaders = imagenet_distribute(data_dir=args.data_dir,
                                      img_size=config['encoder']['img_size'],
                                      batch_size=config['training']['batch_size'])

    model = create_gdt_cls(
        img_size=config['encoder']['img_size'],
        stages_config=config['encoder']['stages'],
        target_leaf_size=config['classifier']['target_leaf_size'],
        encoder_embed_dim=config['encoder']['embed_dim'],
        classifier_embed_dim=config['classifier']['embed_dim'],
        num_classes=config['classifier']['num_classes']
    )
    
    # --- Load checkpoint if specified ---
    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    if args.reload and os.path.exists(checkpoint_path):
        logging.info(f"Reloading model from {checkpoint_path}")
        # Load state dict on the correct device
        map_location = {'cuda:0': f'cuda:{device_id}'}
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device_id, args, config, is_ddp=True)
    dist.destroy_process_group()
    
def gdt_imagenet_train_local(args, config):
    """Main function for local (non-DDP) training."""
    setup_logging(args)
    
    device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        logging.info(f"Starting local training on device cuda:{device_id}")
    else:
        device_id = "cpu"
        logging.info(f"Starting local training on device {device_id}")

    # Update args with config for dataloader
    args.img_size = config['encoder']['img_size']
    args.batch_size = config['training']['batch_size']
    dataloaders, _ = imagenet_subloaders(subset_data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = create_gdt_cls(
        img_size=config['encoder']['img_size'],
        stages_config=config['encoder']['stages'],
        target_leaf_size=config['classifier']['target_leaf_size'],
        encoder_embed_dim=config['encoder']['embed_dim'],
        classifier_embed_dim=config['classifier']['embed_dim'],
        num_classes=config['classifier']['num_classes']
    )

    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    if args.reload and os.path.exists(checkpoint_path):
        logging.info(f"Reloading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device_id)))

    model.to(device_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    # --- FIXED: No need for MockSampler or MockDDP ---
    # The train_model function now handles both cases via the `is_ddp` flag.
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device_id, args, config, is_ddp=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDT-ViT Training Script")
    
    parser.add_argument('--config', type=str, default='./configs/gdt_vit_root_test.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='gdt_vit_root_test', help='Subdirectory for saving logs and models')
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    parser.add_argument('--local', action='store_true', help='Run training locally without DDP')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update args from config for consistency
    args.img_size = config['encoder']['img_size']
    args.num_epochs = config['training']['num_epochs']
    args.batch_size = config['training']['batch_size']
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    gdt_imagenet_train_local(args, config)