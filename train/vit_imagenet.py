import os
import sys
import yaml
import logging
import argparse
import contextlib
import torch.compiler

import torch
from torch import nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast 
from torch.profiler import profile, record_function, ProfilerActivity

# Import the baseline ViT model and the dataset functions
sys.path.append("./")
from dataset.imagenet import imagenet_distribute, imagenet_subloaders
# from model.vit import create_vit_model
from model.vit import create_timm_vit as create_vit_model
# from model.vit import create_rope_vit_model as create_vit_model


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

def train_vit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, device_id, args, start_epoch, best_val_acc=0.0, is_ddp=False):
    
    # *** 移除 SCALER ***
    # GradScaler 仅用于 float16，不用于 bfloat16
    # scaler = GradScaler(enabled=True) 
    
    num_epochs = config['training']['num_epochs']
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        label_smoothing=0.1,
        prob=1.0,              # 总使用概率
        switch_prob=0.5        # mixup vs cutmix 切换概率
    )
    
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info("Starting ViT training for %d epochs with Automatic Mixed Precision (AMP)...", num_epochs)   
        logging.info("开始BF16优化训练 (注意: GradScaler 已为 BF16 禁用)...") # 更新日志
        logging.info(f"torch.compile: {config['training'].get('use_compile', False)}, Fused Optimizer: {config['training'].get('use_fused_optimizer', False)}, Activation Checkpointing: {config['training'].get('use_checkpointing', False)}")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
        
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            
            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            original_labels = labels.clone()
            
            if config['training']['use_mixup']:
                images, soft_labels = mixup_fn(images, labels)
            else:
                soft_labels = nn.functional.one_hot(labels, config['model']['num_classes']).float()
                
            sync_context = model.no_sync() if (is_ddp and is_accumulation_step) else contextlib.nullcontext()
            
            with sync_context:
                
                # *** 修复: 将 mark_step_begin 移动到 *内部* ***
                torch.compiler.cudagraph_mark_step_begin()
                
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    
                    # *** 修复 (1/2): 立即克隆 outputs 以进行后续评估 ***
                    # 更改: .data.clone() -> .clone().detach() (更安全)
                    outputs_for_eval = outputs.clone().detach()
                    
                    loss = criterion(outputs, soft_labels)
                    loss = loss / accumulation_steps
                
                loss.backward()

            if not is_accumulation_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and not config['training']['use_postrain']: scheduler.step()
                
            # --- 之后的评估逻辑保持不变 ---
            
            # *** 修复 (2/2): 使用克隆的张量进行评估 ***
            _, predicted = torch.max(outputs_for_eval, 1)
            running_total += original_labels.size(0)
            running_corrects += (predicted == original_labels).sum().item()
            running_loss += loss.item() * accumulation_steps

            if (i + 1) % 10 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, current_lr: {current_lr:.6f}')
                running_loss, running_corrects, running_total = 0.0, 0, 0

        val_acc = evaluate_model_compatible(
            model, 
            val_loader, 
            device_id, 
            is_ddp=is_ddp
        )
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f} | current_lr: {current_lr:.6f}")
            
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
                logging.info(f"新的最佳验证精度: {best_val_acc:.4f}. 保存最佳模型...")
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
    """评估函数。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # 验证时继续使用 bfloat16
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for images, labels in val_loader:
                
                # *** 修复: 为验证循环添加 mark_step_begin ***
                # CUDAGraphs 也会捕获评估循环
                torch.compiler.cudagraph_mark_step_begin()
                
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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def vit_imagenet_train(args, config):
    """(SLURM) Main DDP training function for the baseline ViT."""
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup_ddp(rank, world_size)
    
    local_rank = int(os.environ['SLURM_LOCALID'])
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    if rank == 0: setup_logging(args)
    logging.info(f"DDP Initialized for ViT Baseline. Rank {rank}/{world_size} on device {device_id}")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    
    dataloaders = imagenet_distribute(
    img_size=args.img_size,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers)

    model = create_vit_model(config)
        
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    model_without_ddp = model.module
    param_groups = param_groups_lrd(model_without_ddp, config['training']['weight_decay'],
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0.75
    )
    
    use_fused = config['training'].get('use_fused_optimizer', False)
    optimizer = torch.optim.AdamW(
            param_groups, 
            lr=config['training']['learning_rate'], 
            weight_decay=config['training']['weight_decay'],
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
            logging.info(f"从检查点恢复训练: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        if dist.get_rank() == 0:
            logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。")
            
    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device_id, args, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=(world_size > 1))
    dist.destroy_process_group()

def vit_imagenet_train_single(args, config):
    # (torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = 'nccl'
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"开始训练，使用 {world_size} 个进程，设备类型: {device.type}")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    
    dataloaders = imagenet_distribute(
    img_size=args.img_size,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers)
    
    model = create_vit_model(config).to(device)
        
    if config['training'].get('use_compile', False):
        if dist.get_rank() == 0: logging.info("正在应用 torch.compile()...")
        
        # *** 修复: 从 'max-autotune' 回退到 'default' ***
        # 'max-autotune' 启用 CUDAGraphs，这导致了内存覆盖错误.
        # 'default' 仍然会融合 FA2，但对 CUDAGraphs 不那么激进.
        model = torch.compile(model, mode="default")
        
    if device.type == 'cuda' or world_size > 1:
        if device.type == 'cuda':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    criterion = nn.CrossEntropyLoss()
    model_without_ddp = model.module
    param_groups = param_groups_lrd(model_without_ddp, config['training']['weight_decay'],
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0.65
    )
    use_fused = config['training'].get('use_fused_optimizer', False)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', (0.9, 0.95))),
        fused=use_fused,
        amsgrad=False
    )
    
    training_config = config['training']
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 0) 
    accumulation_steps = training_config.get('gradient_accumulation_steps', 1) 
    optimizer_steps_per_epoch = len(dataloaders['train']) // accumulation_steps

    num_training_steps = num_epochs * optimizer_steps_per_epoch
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch

    
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
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
            logging.info(f"从检查点恢复训练: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        
        if dist.get_rank() == 0:
            logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。")
            
    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config, device, args, start_epoch=start_epoch, best_val_acc=best_val_acc, is_ddp=(world_size > 1))
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Training Script")
    
    parser.add_argument('--config', type=str, default='./configs/vit-b_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='vit-b-16-he-timm-pz4-fa2', help='Subdirectory for saving logs and models')
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    parser.add_argument('--local', action='store_true', help='Run training locally without DDP')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args.img_size = config['model']['img_size']
    args.num_epochs = config['training']['num_epochs']
    args.batch_size = config['training']['batch_size']
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    vit_imagenet_train_single(args, config)

