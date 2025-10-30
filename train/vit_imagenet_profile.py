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

# *** 导入 Profiler ***
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

def train_vit_model(model, train_loader, criterion, optimizer, scheduler, config, device_id, args, is_ddp=False):
    """
    *** 简化的训练函数，用于性能分析 ***
    此函数仅运行几个步骤来收集性能数据。
    """
    
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
        logging.info("Starting ViT PROFILING run...")
        logging.info("开始BF16优化训练 (注意: GradScaler 已为 BF16 禁用)...")
        logging.info(f"torch.compile: {config['training'].get('use_compile', False)}, Fused Optimizer: {config['training'].get('use_fused_optimizer', False)}")
        
    model.train()
    
    # Profiler 设置: 1=Wait, 2=Warmup, 5=Active, 1=Repeat
    wait_steps = 1
    warmup_steps = 2
    active_steps = 5
    total_steps = wait_steps + warmup_steps + active_steps
    
    if is_main_process:
        logging.info(f"Profiler 将运行 {total_steps} 个步骤 (Wait={wait_steps}, Warmup={warmup_steps}, Active={active_steps})...")

    # 获取数据加载器的迭代器
    data_iter = iter(train_loader) 

    # *** 启动 Profiler ***
    profiler_output_dir = os.path.join(args.output, args.savefile, "profiler_trace")
    if is_main_process: os.makedirs(profiler_output_dir, exist_ok=True)
        
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_output_dir),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        
        for i in range(total_steps):
            if is_ddp: train_loader.sampler.set_epoch(0) # 保持 epoch 0
            
            try:
                images, labels = next(data_iter)
            except StopIteration:
                if is_main_process: logging.warning("DataLoader 在 profiling 期间耗尽. 重启中...")
                data_iter = iter(train_loader)
                images, labels = next(data_iter)

            is_accumulation_step = (i + 1) % accumulation_steps != 0
            images = images.to(device_id, non_blocking=True)
            labels = labels.to(device_id, non_blocking=True)
            
            if config['training']['use_mixup']:
                images, soft_labels = mixup_fn(images, labels)
            else:
                soft_labels = nn.functional.one_hot(labels, config['model']['num_classes']).float()
                
            sync_context = model.no_sync() if (is_ddp and is_accumulation_step) else contextlib.nullcontext()
            
            with sync_context:
                torch.compiler.cudagraph_mark_step_begin()
                
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(images)
                    # 注意: 我们跳过了 clone() 和评估，因为我们只关心前向+后向的性能
                    loss = criterion(outputs, soft_labels)
                    loss = loss / accumulation_steps
                
                loss.backward()

            if not is_accumulation_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler: scheduler.step() # 保持调度器同步
            
            if is_main_process:
                 logging.info(f"[Step {i+1}/{total_steps}] Profiling step complete. Loss: {loss.item() * accumulation_steps:.3f}")

            # *** 通知 profiler 步骤完成 ***
            prof.step()

    if is_main_process:
        logging.info("--- [Profiler] 性能分析完成 ---")
        # 打印按 CUDA 时间排序的顶部内核
        print("\n--- Top 20 CUDA Kernels (by Total Time) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # 打印按 CPU 时间排序的顶部操作
        print("\n--- Top 20 CPU Operators (by Total Time) ---")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        
        # 打印内存使用情况
        print("\n--- Top 10 Memory Usage Operators ---")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        
        logging.info(f"Profiler trace (用于 TensorBoard) 已保存到: {profiler_output_dir}")
        logging.info("--- [Profiler] 运行结束 ---")

# (evaluate_model_compatible, setup_ddp, vit_imagenet_train 保持不变，但不会被调用)
# ... (为简洁起见，省略 evaluate_model_compatible, setup_ddp, vit_imagenet_train)

def evaluate_model_compatible(model, val_loader, device, is_ddp=False):
    """评估函数。(在此脚本中不使用)"""
    model.eval()
    # ... (代码不变) ...

def setup_ddp(rank, world_size):
    """(在此脚本中不使用)"""
    # ... (代码不变) ...
    
def vit_imagenet_train(args, config):
    """(在此脚本中不使用)"""
    # ... (代码不变) ...


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
        logging.info(f"开始 PROFILING 运行，使用 {world_size} 个进程，设备类型: {device.type}")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    
    dataloaders = imagenet_distribute(
        img_size=args.img_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
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
    
    # 调度器 (对于profiling不是必需的，但保持以使图一致)
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
    
    # *** 简化: 移除检查点加载逻辑 ***
    # ...
            
    # *** 调用简化的 profiling 函数 ***
    train_vit_model(
        model, 
        dataloaders['train'], 
        criterion, 
        optimizer, 
        scheduler, 
        config, 
        device, 
        args, 
        is_ddp=(world_size > 1)
    )
    
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Training Script")
    
    parser.add_argument('--config', type=str, default='./configs/vit-b_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    # *** 修改 savefile 以反映 profiling ***
    parser.add_argument('--savefile', type=str, default='vit-b-16-profile', help='Subdirectory for saving logs and models')
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader')
    # *** 简化: 移除 --reload 和 --local ***
    # parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists')
    # parser.add_argument('--local', action='store_true', help='Run training locally without DDP')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 更新 args (保持不变)
    args.img_size = config['model']['img_size']
    args.num_epochs = config['training']['num_epochs']
    args.batch_size = config['training']['batch_size']
    
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    vit_imagenet_train_single(args, config)
