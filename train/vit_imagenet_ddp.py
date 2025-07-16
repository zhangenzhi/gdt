import os
import sys
import yaml
import logging
import argparse

import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast 


# Import the baseline ViT model and the dataset functions
sys.path.append("./")
from dataset.imagenet import imagenet_distribute, imagenet_subloaders
from model.vit import create_vit_model

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

def train_vit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_id, args, is_ddp=False):
    # 1. 初始化 GradScaler
    # enabled=True 表示启用混合精度。可以将其设为命令行参数来控制是否开启。
    scaler = GradScaler(device_type='cuda', enabled=True)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        # 在日志中注明正在使用混合精度
        logging.info("Starting ViT training for %d epochs with Automatic Mixed Precision (AMP)...", num_epochs)

    for epoch in range(num_epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 2. 使用 autocast 上下文管理器包裹前向传播和损失计算
            # dtype=torch.bfloat16 是H100的最佳选择。也可以使用 torch.float16
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 3. 使用 GradScaler 缩放损失并执行反向传播
            # scaler.scale(loss) 会将损失乘以一个缩放因子
            scaler.scale(loss).backward()

            # 4. scaler.step() 会先将梯度反缩放，然后调用优化器执行更新
            scaler.step(optimizer)

            # 5. 更新 GradScaler 的缩放因子
            scaler.update()

            # --- 之后的评估逻辑保持不变 ---
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            # 注意：loss.item() 会自动返回未缩放的、float32类型的损失值
            running_loss += loss.item()

            if (i + 1) % 10 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 10
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%')
                running_loss, running_corrects, running_total = 0.0, 0, 0
                
        scheduler.step()

        # 从您的模型配置中获取use_fp8标志
        model_config = config.get('model', {})
        use_fp8_flag = model_config.get('use_fp8', False)

        # 调用兼容性更强的评估函数
        val_acc = evaluate_model_compatible(
            model, 
            val_loader, 
            device_id, 
            is_ddp=is_ddp, 
            use_fp8=use_fp8_flag
        )
                
        if is_main_process:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {best_val_acc:.4f}. Saving model to {checkpoint_path}")
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(), checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Training. Best Validation Accuracy: {best_val_acc:.4f}')

        
# 确保导入了 transformer_engine
try:
    import transformer_engine.pytorch as te
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False

def evaluate_model_compatible(model, val_loader, device_id, is_ddp=False, use_fp8=False):
    """
    一个兼容标准、混合精度和FP8模型的评估函数。
    """
    model.eval()
    correct = 0
    total = 0
    
    # 在评估时禁用梯度计算，以节省显存和计算
    with torch.no_grad():
        # 根据是否使用FP8，选择不同的autocast上下文
        autocast_context = None
        if use_fp8:
            if not TRANSFORMER_ENGINE_AVAILABLE:
                raise ImportError("请求使用FP8进行评估，但Transformer Engine未安装。")
            # 使用Transformer Engine的FP8上下文
            autocast_context = te.fp8_autocast(enabled=True)
        else:
            # 对于标准或BF16/FP16模型，使用PyTorch原生的autocast
            # 注意: 这里的dtype应该与训练时匹配，例如 torch.bfloat16
            autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True)

        for images, labels in val_loader:
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)

            with autocast_context:
                outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 在DDP模式下，聚合所有进程的结果
    if is_ddp:
        # 将Python标量转换为PyTorch张量，并放到当前GPU上
        total_tensor = torch.tensor(total, device=device_id)
        correct_tensor = torch.tensor(correct, device=device_id)
        
        # 使用 all_reduce 操作对所有GPU上的张量求和
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
        # 将聚合后的结果从张量转回Python标量
        total = total_tensor.item()
        correct = correct_tensor.item()
    
    # 计算最终精度，只有主进程需要这个结果，但这里为简单起见所有进程都计算
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
def vit_imagenet_train(args, config):
    """Main DDP training function for the baseline ViT."""
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
    dataloaders = imagenet_distribute(args=args)

    model = create_vit_model(config)
    
    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    if args.reload and os.path.exists(checkpoint_path):
        logging.info(f"Reloading model from {checkpoint_path}")
        map_location = {'cuda:0': f'cuda:{device_id}'}
        model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
        
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device_id, args, is_ddp=True)
    dist.destroy_process_group()

def vit_imagenet_train_local(args, config):
    """Main function for local (non-DDP) baseline ViT training."""
    setup_logging(args)
    
    device_id = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        logging.info(f"Starting local training on device cuda:{device_id}")
    else:
        device_id = "cpu"
        logging.info(f"Starting local training on device {device_id}")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    dataloaders, _ = imagenet_subloaders(subset_data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = create_vit_model(config)

    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    if args.reload and os.path.exists(checkpoint_path):
        logging.info(f"Reloading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device_id)))

    model.to(device_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])

    train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device_id, args, is_ddp=False)

def train_on_single(args, config):
    """
    一个更健壮的主函数，能自动适应多卡GPU、单卡GPU和纯CPU环境。
    """
    # torchrun 会自动设置 'LOCAL_RANK' 等环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # --- 关键修改：根据环境选择后端和设备 ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        backend = 'nccl'  # 使用GPU
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        backend = 'gloo'  # 使用CPU
        device = torch.device("cpu")
    
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)
    # --- 修改结束 ---

    # 只有主进程 (rank 0) 才执行日志设置和打印
    if local_rank == 0:
        setup_logging(args)
        logging.info(f"开始训练，使用 {world_size} 个进程，设备类型: {device.type}")

    # ... 之后的代码（数据加载、模型创建等）与之前基本相同 ...
    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    
    # 注意：imagenet_distribute 内部也需要能处理 device='cpu' 的情况
    dataloaders = imagenet_distribute(
    img_size=args.img_size,
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers)
    
    model = create_vit_model(config).to(device)
    
    # 只有在GPU上才需要用DDP包装。在CPU上，当world_size > 1时也需要
    if device.type == 'cuda' or world_size > 1:
        # 对于CPU的多进程训练，也需要DDP
        # 注意：在CPU模式下，DDP不需要 device_ids 参数
        if device.type == 'cuda':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)

    # ... 之后的代码（优化器、训练循环等）完全相同 ...
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'], eta_min=config['training']['min_lr'])
    
    # train_vit_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device, args, is_ddp=(world_size > 1))
    train_vit_model_fp8(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, scheduler, config['training']['num_epochs'], device, args, is_ddp=(world_size > 1))
    dist.destroy_process_group()

# ==============================================================================
# FP8 优化的训练函数
# ==============================================================================
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
def train_vit_model_fp8(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device_id, args, is_ddp=False):
    """
    为FP8优化的ViT模型设计的主训练循环。
    """
    # 1. 不再需要 GradScaler
    # scaler = GradScaler(...)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.output, args.savefile, "best_model.pth")
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info("开始ViT训练，共 %d 个epoch (使用FP8优化)...", num_epochs)

    # 2. 定义FP8训练的"配方"
    #    E4M3是前向传播的格式, HYBRID是反向传播的格式, 这是H100的推荐默认值。
    fp8_recipe = recipe.DelayedScaling(
        margin=0, interval=1, fp8_format=recipe.Format.E4M3
    )

    for epoch in range(num_epochs):
        if is_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        running_loss, running_corrects, running_total = 0.0, 0, 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            # 恢复标准的梯度清零
            optimizer.zero_grad()
            
            # 3. 使用 Transformer Engine 的 fp8_autocast 上下文管理器
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 4. 恢复标准的后向传播和优化器更新
            loss.backward()
            optimizer.step()

            # --- 之后的评估逻辑保持不变 ---
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i + 1) % 10 == 0 and is_main_process:
                train_acc = 100 * running_corrects / running_total if running_total > 0 else 0
                avg_loss = running_loss / 10
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}]  Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%')
                running_loss, running_corrects, running_total = 0.0, 0, 0
                
        scheduler.step()
        
        # 注意：您的评估函数(evaluate_baseline_model)内部也应该使用 fp8_autocast
        # 以确保数据类型一致并获得加速，但评估时不需要梯度计算。
                # 从您的模型配置中获取use_fp8标志
        model_config = config.get('model', {})
        use_fp8_flag = model_config.get('use_fp8', False)

        # 调用兼容性更强的评估函数
        val_acc = evaluate_model_compatible(
            model, 
            val_loader, 
            device_id, 
            is_ddp=is_ddp, 
            use_fp8=use_fp8_flag
        )
        
        if is_main_process:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f"新的最佳验证精度: {best_val_acc:.4f}. 保存模型至 {checkpoint_path}")
                model_to_save = model.module if is_ddp else model
                torch.save(model_to_save.state_dict(), checkpoint_path)

    if is_main_process:
        logging.info(f'训练完成。最佳验证精度: {best_val_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Training Script")
    
    parser.add_argument('--config', type=str, default='./configs/vit_test.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    # parser.add_argument('--savefile', type=str, default='vit-b-16', help='Subdirectory for saving logs and models')
    parser.add_argument('--savefile', type=str, default='vit-b-16-fp8', help='Subdirectory for saving logs and models')
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    # Use action='store_true' for boolean flags
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    parser.add_argument('--local', action='store_true', help='Run training locally without DDP')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Update args from config for consistency
    args.img_size = config['model']['img_size']
    args.num_epochs = config['training']['num_epochs']
    args.batch_size = config['training']['batch_size']
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    train_on_single(args, config)