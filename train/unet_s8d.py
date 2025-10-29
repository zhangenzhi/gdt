import os
import sys
# 确保可以找到自定义模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import logging
import argparse
import contextlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from model.unet import create_unet_model # 复用之前的 U-Net 工厂函数
# 使用新的多类别损失函数
from model.losses import DiceCELossMulti, mean_dice_score_multi
# 使用新的 s8d 数据加载器
from dataset.s8d import build_s8d_segmentation_dataloaders


def setup_logging(output_dir, save_dir):
    """配置日志记录到文件和控制台。"""
    log_dir = os.path.join(output_dir, save_dir)
    os.makedirs(log_dir, exist_ok=True)
    rank = dist.get_rank() if dist.is_initialized() else 0
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_color_map(num_classes=5):
    # 定义一个简单的颜色映射 (可自定义)
    # 0: 背景 (黑色), 1: 红色, 2: 绿色, 3: 蓝色, 4: 黄色
    colors = torch.tensor([
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
    ], dtype=torch.uint8)
    if num_classes > len(colors):
        # 如果类别数超过预定义颜色，生成随机颜色
        extra_colors = torch.randint(0, 256, (num_classes - len(colors), 3), dtype=torch.uint8)
        colors = torch.cat([colors, extra_colors], dim=0)
    return colors[:num_classes]

def visualize_predictions(epoch, output_dir, save_dir, images, targets, preds_logits, val_loader, num_classes=5):
    """可视化验证集上的预测结果（多类别）。"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if not is_main_process: return # 仅主进程执行

    color_map = get_color_map(num_classes).cpu().numpy()

    # 反归一化
    mean = val_loader.mean if hasattr(val_loader, 'mean') else 0.0
    std = val_loader.std if hasattr(val_loader, 'std') else 1.0

    # Ensure tensors are on the correct device for visualization (usually CPU)
    images = images.cpu()
    targets = targets.cpu()
    preds_logits = preds_logits.cpu()


    mean_tensor = torch.tensor([mean] * images.shape[1]).view(1, -1, 1, 1) # No need for device here
    std_tensor = torch.tensor([std] * images.shape[1]).view(1, -1, 1, 1)

    fig, axes = plt.subplots(3, 7, figsize=(28, 12))
    fig.suptitle(f'Epoch {epoch + 1} Validation Results (U-Net S8D)', fontsize=20)

    num_samples_to_show = min(7, images.shape[0])

    # 获取预测类别
    preds = preds_logits.argmax(dim=1) # (N, H, W)

    for i in range(num_samples_to_show):
        img_tensor = images[i]
        # Perform denormalization on CPU
        img_denorm = img_tensor * std_tensor.squeeze() + mean_tensor.squeeze()
        img_denorm = img_denorm.clamp(0, 1)


        img_np = img_denorm.numpy() # Already on CPU
        if img_np.shape[0] == 1:
            img_np_display = img_np.squeeze(0)
            cmap_img = 'gray'
        else:
            img_np_display = img_np.transpose(1, 2, 0)
            cmap_img = None

        tgt = targets[i].numpy() # (H, W)
        prd = preds[i].numpy()   # (H, W)

        # 应用颜色映射
        # Handle potential out-of-bounds indices if target has unexpected values
        tgt_clipped = np.clip(tgt, 0, num_classes - 1)
        prd_clipped = np.clip(prd, 0, num_classes - 1)
        tgt_color = color_map[tgt_clipped] # (H, W, 3)
        prd_color = color_map[prd_clipped] # (H, W, 3)


        axes[0, i].imshow(img_np_display, cmap=cmap_img)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(tgt_color)
        axes[1, i].set_title(f'Ground Truth (Colored)')
        axes[1, i].axis('off')

        axes[2, i].imshow(prd_color)
        axes[2, i].set_title(f'Prediction (Colored)')
        axes[2, i].axis('off')

    for i in range(num_samples_to_show, 7):
        axes[0, i].axis('off'); axes[1, i].axis('off'); axes[2, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, save_dir, f"validation_epoch_{epoch+1}.png")
    try:
        plt.savefig(save_path)
        plt.close()
        logging.info(f"验证结果可视化已保存至: {save_path}")
    except Exception as e:
         logging.error(f"保存可视化图像失败: {e}")
         plt.close() # Ensure figure is closed even on error


def evaluate(model, val_loader, criterion, device, epoch, args, num_classes):
    """评估模型（多类别）并可视化结果。不进行 DDP 同步。"""
    model.eval()
    total_val_loss = 0.0
    total_dice_score = 0.0
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    all_images, all_targets, all_logits = [], [], [] # 仅在主进程填充

    with torch.no_grad():
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(not is_main_process))
        for images, targets, _ in val_iterator: # 解包 slice_id
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long() # 确保是 Long

            try:
                with autocast(dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits.float(), targets) # 全精度计算损失

                batch_loss = loss.item()
                # 计算平均前景 Dice
                batch_dice = mean_dice_score_multi(logits, targets, num_classes=num_classes, exclude_background=True)

                # --- **修改点: 移除 DDP 同步 Batch 指标** ---
                # if dist.is_initialized():
                #     loss_tensor = torch.tensor(batch_loss, device=device)
                #     dice_tensor = torch.tensor(batch_dice, device=device)
                #     dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                #     dist.all_reduce(dice_tensor, op=dist.ReduceOp.AVG)
                #     batch_loss = loss_tensor.item()
                #     batch_dice = dice_tensor.item()

                total_val_loss += batch_loss
                total_dice_score += batch_dice

                if is_main_process:
                    # 仅收集第一个 batch 的数据用于可视化，以防验证集过大
                    if len(all_images) == 0:
                         # Move tensors to CPU before appending to avoid GPU memory accumulation
                         all_images.append(images.cpu())
                         all_targets.append(targets.cpu())
                         all_logits.append(logits.cpu()) # 保存 logits 用于可视化

                    val_iterator.set_postfix({'val_loss': f"{batch_loss:.4f}", 'val_dice': f"{batch_dice:.4f}"})

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if is_main_process:
                         logging.error(f"验证期间 CUDA 内存不足！跳过此 batch。 Batch shape: {images.shape}")
                    torch.cuda.empty_cache() # 尝试释放内存
                    continue # 跳过这个 batch
                else:
                    if is_main_process:
                         logging.error(f"验证期间发生错误: {e}", exc_info=True)
                    # Don't re-raise during eval, just log and continue if possible
                    continue # Skip batch on other errors during eval


    # 计算本地进程的平均值
    num_batches = len(val_loader)
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_score = total_dice_score / num_batches if num_batches > 0 else 0.0

    # 主进程执行可视化
    if is_main_process:
        if all_images and all_targets and all_logits:
            visualize_predictions(
                epoch, args.output, args.savefile,
                torch.cat(all_images)[:7], torch.cat(all_targets)[:7], torch.cat(all_logits)[:7],
                val_loader, num_classes=num_classes
            )
        else:
            logging.warning("未能收集到用于可视化的验证数据。")

    # 返回本地进程的平均值
    return avg_val_loss, avg_dice_score


def train(config, args):
    """主训练函数"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if use_ddp:
        # 增加超时以防初始化卡住
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=7200))


    is_main_process = (local_rank == 0)
    if is_main_process:
        setup_logging(args.output, args.savefile)
        logging.info(f"开始 U-Net S8D 训练，使用 {world_size} 个进程...") # 更新日志
        logging.info(f"配置: {config}")
        logging.info(f"参数: {args}")

    # --- 数据加载 ---
    if is_main_process: logging.info("正在加载 S8D 数据集...")
    try:
        dataloaders = build_s8d_segmentation_dataloaders(
            data_dir=args.data_dir,
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            # img_size=config['model']['img_size'], # 传递 img_size
            # use_ddp=use_ddp
        )
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
    except Exception as e:
         if is_main_process: logging.error(f"加载数据时出错: {e}", exc_info=True)
         if use_ddp: dist.destroy_process_group()
         sys.exit(1)

    # --- 模型创建 ---
    if is_main_process: logging.info(f"创建模型: U-Net with {config['model']['backbone_name']} backbone")
    # 复用 unet.py 中的 create_unet_model
    model = create_unet_model(
        backbone_name=config['model']['backbone_name'],
        pretrained=config['model']['pretrained'],
        in_chans=config['model']['in_chans'],
        num_classes=config['model']['num_classes'] # 设置多类别输出
    ).to(device)

    # 编译和 DDP
    if config['training'].get('use_compile', False):
        if is_main_process: logging.info("应用 torch.compile()...")
        model = torch.compile(model, mode="max-autotune")
    if use_ddp:
        # 对于标准 U-Net，通常不需要 find_unused_parameters=True
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # --- 损失，优化器，调度器 ---
    num_classes = config['model']['num_classes']
    criterion = DiceCELossMulti(num_classes=num_classes) # 使用多类别损失
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training']['betas'])
    )

    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    # 计算每个进程的 optimizer 步数
    local_steps_per_epoch = len(train_loader) // accumulation_steps if len(train_loader) > 0 else 0
    # 全局总步数（用于调度器）= 每个进程的步数 (因为 scheduler.step() 在每个进程都调用)
    optimizer_steps_per_epoch = local_steps_per_epoch

    num_training_steps = num_epochs * optimizer_steps_per_epoch
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch

    if is_main_process:
        logging.info(f"总 Epochs: {num_epochs}, 每个进程 Epoch 优化器步数: {optimizer_steps_per_epoch}")
        logging.info(f"总步数 (用于调度器): {num_training_steps}, Warmup 步数: {num_warmup_steps}")

    if optimizer_steps_per_epoch == 0:
        if is_main_process:
            logging.warning("每个 epoch 的优化器步数为 0 (可能是 dataloader 为空或 accumulation_steps 过大)。跳过调度器设置。")
        scheduler = None # 没有步数，无法设置调度器
    elif num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps - num_warmup_steps), eta_min=config['training']['eta_min'])
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps), eta_min=config['training']['eta_min'])

    scaler = GradScaler(enabled=True)

    # --- 检查点加载 ---
    start_epoch = 0
    best_dice = 0.0 # 使用 Dice 作为标准
    checkpoint_dir = os.path.join(args.output, args.savefile)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if args.reload and os.path.exists(latest_checkpoint_path):
        if is_main_process: logging.info(f"从检查点恢复: {latest_checkpoint_path}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
            model_to_load = model.module if use_ddp else model
            missing, unexpected = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if is_main_process:
                 if missing: logging.warning(f"加载模型权重时缺失: {missing}")
                 if unexpected: logging.warning(f"加载模型权重时意外: {unexpected}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 只有在调度器有效时才加载
            if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint: scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            if is_main_process: logging.info(f"成功恢复，从 Epoch {start_epoch + 1} 开始。最佳 Dice: {best_dice:.4f}")
        except Exception as e:
            if is_main_process: logging.error(f"加载检查点失败: {e}。从头开始。")
            start_epoch = 0; best_dice = 0.0

    # --- 训练循环 ---
    if is_main_process: logging.info(f"开始训练，从 Epoch {start_epoch + 1} 到 {num_epochs}...")

    # 在循环外初始化 epoch 累积损失 (仅本地)
    # epoch_losses = [] # 不再需要

    for epoch in range(start_epoch, num_epochs):
        if use_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        running_loss_local = 0.0 # 本地累加损失
        num_local_steps = 0      # 本地实际执行的step数

        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", disable=(not is_main_process))

        for i, (images, targets, _) in enumerate(train_iterator): # 解包 slice_id
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long() # 确保是 Long

            is_sync_step = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)
            # DDP forward pass should happen outside no_sync context if is_sync_step is True
            # For simplicity, keep contextlib.nullcontext() when DDP is off or it's sync step
            sync_context = contextlib.nullcontext() if not use_ddp or is_sync_step else model.no_sync()


            try:
                # Forward pass outside no_sync context if is_sync_step
                with autocast(dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits.float(), targets) # 全精度计算损失
                    loss = loss / accumulation_steps

                # Backward pass might need to be within sync_context if not is_sync_step
                # However, the example code does backward outside. Let's stick to that for now.
                # If DDP issues arise (e.g., param not receiving grad), adjust this.
                scaler.scale(loss).backward()


                if is_sync_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step() # 只有在 scheduler 有效时才 step

                    # 累加本地 batch 损失 (未标准化的)
                    batch_loss_local = loss.item() * accumulation_steps
                    running_loss_local += batch_loss_local
                    num_local_steps += 1

                    # 更新 tqdm (仅主进程)
                    if is_main_process:
                        current_avg_loss_local = running_loss_local / num_local_steps if num_local_steps > 0 else 0.0
                        train_iterator.set_postfix({
                            'avg_loss (local)': f"{current_avg_loss_local:.4f}", # 标记为本地损失
                            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                        })

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e):
                     if is_main_process:
                         logging.error(f"训练期间 CUDA 内存不足！Epoch {epoch+1}, Batch {i+1}. Batch shape: {images.shape}. 尝试跳过。")
                     torch.cuda.empty_cache()
                     optimizer.zero_grad(set_to_none=True)
                     continue
                 else:
                     if is_main_process:
                         logging.error(f"训练期间发生错误: {e}", exc_info=True)
                     # 移除 barrier
                     raise e # 重新抛出错误

        # --- 计算 Epoch 平均训练损失 (仅本地) ---
        avg_train_loss_local = running_loss_local / num_local_steps if num_local_steps > 0 else 0.0


        # --- 评估 ---
        # 移除 barrier
        try:
            val_loss, val_dice = evaluate(model, val_loader, criterion, device, epoch, args, num_classes)
        except Exception as e:
             if is_main_process:
                 logging.error(f"评估 Epoch {epoch+1} 时出错: {e}", exc_info=True)
             val_loss, val_dice = float('inf'), 0.0 # 设置默认值以继续流程


        # --- 日志与检查点 (仅主进程) ---
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else 0.0
            # 使用 evaluate 返回的本地 rank 0 的结果
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Avg Train Loss (Rank 0): {avg_train_loss_local:.4f} | "
                f"Val Loss (Rank 0): {val_loss:.4f} | " # 标记为 Rank 0
                f"Val Dice (Rank 0, avg fg): {val_dice:.4f} | " # 标记为 Rank 0
                f"LR: {current_lr:.6f}"
            )

            # 保存最新检查点
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': (model.module if use_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None, # 保存 scheduler 状态
                'scaler_state_dict': scaler.state_dict(),
                'best_dice': best_dice,
                'config': config,
                'args': vars(args)
            }
            try:
                torch.save(checkpoint_data, latest_checkpoint_path)
            except Exception as e:
                logging.error(f"保存最新检查点失败: {e}")

            # 保存最佳模型 (基于 Rank 0 的 val_dice)
            if val_dice > best_dice:
                best_dice = val_dice
                logging.info(f"新最佳模型 (Rank 0)，Dice: {best_dice:.4f}. 保存中...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                try:
                    torch.save((model.module if use_ddp else model).state_dict(), best_checkpoint_path)
                except Exception as e:
                     logging.error(f"保存最佳模型失败: {e}")

    # --- 训练结束 ---
    if is_main_process:
        logging.info(f'训练完成. 最佳验证 Dice (Rank 0, avg fg): {best_dice:.4f}')

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net S8D Segmentation Training Script")
    parser.add_argument('--config', type=str, default='./configs/unet-resnet_S8D.yaml', help='配置文件路径') # 默认 U-Net S8D 配置
    parser.add_argument('--data_dir', type=str, default=None, help='数据集目录 (覆盖配置)')
    parser.add_argument('--output', type=str, default=None, help='输出根目录 (覆盖配置)')
    parser.add_argument('--savefile', type=str, default=None, help='保存文件夹名 (覆盖配置)')
    parser.add_argument('--reload', action='store_true', help='从检查点恢复')

    args = parser.parse_args()

    # 加载配置
    try:
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
    except FileNotFoundError: print(f"错误: 配置文件 '{args.config}' 未找到。"); sys.exit(1)
    except Exception as e: print(f"加载配置文件 '{args.config}' 出错: {e}"); sys.exit(1)

    # 覆盖配置
    if args.data_dir: config.setdefault('data', {})['data_dir'] = args.data_dir
    if args.output: config.setdefault('output', {})['base_dir'] = args.output
    if args.savefile: config.setdefault('output', {})['save_dir'] = args.savefile
    else:
        config.setdefault('output', {})
        config['output']['save_dir'] = config['output'].get('save_dir', config.get('task_name', 'unet_s8d_run'))

    # 设置 args
    args.data_dir = config.get('data', {}).get('data_dir', './data_s8d')
    args.output = config.get('output', {}).get('base_dir', './output_s8d')
    args.savefile = config.get('output', {}).get('save_dir', 'unet_s8d_run')

    # 检查配置
    if 'model' not in config or 'img_size' not in config['model'] or 'num_classes' not in config['model']:
        print("错误: 配置文件缺少 'model' 或 'model.img_size' 或 'model.num_classes'。"); sys.exit(1)
    if not args.data_dir or not os.path.exists(args.data_dir):
         print(f"错误: 数据目录 '{args.data_dir}' 不存在。请通过 --data_dir 或配置文件指定。"); sys.exit(1)


    # 运行训练
    try:
        train(config, args)
    except Exception as e:
         rank = int(os.environ.get("LOCAL_RANK", 0))
         if rank == 0: logging.error(f"训练出错: {e}", exc_info=True)
         if dist.is_initialized(): dist.destroy_process_group()
         sys.exit(1)
