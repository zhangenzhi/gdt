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
import datetime # For DDP timeout

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

# --- 导入我们自己的模块 ---
try:
    # !! 修改: 导入 UNETR 模型 !!
    from model.unetr_model import create_unetr_model
    from model.losses import DiceBCELoss
    from dataset.hdg import create_dataloaders
except ImportError as e:
    print("导入错误: 请确保您的项目文件结构包含 unetr_model.py:")
    print("project_root/")
    print("├── model/")
    print("│   ├── unetr_model.py") # 新增
    print("│   ├── sam_model.py")
    print("│   └── losses.py")
    print("├── dataset/")
    print("│   └── hdg.py")
    print("└── train/")
    print("    ├── unetr_train.py (此脚本)")
    print("    └── sam_seg_train.py")
    print(f"原始错误: {e}")
    sys.exit(1)


def setup_logging(output_dir, save_dir):
    """配置日志记录到文件和控制台。"""
    log_dir = os.path.join(output_dir, save_dir)
    os.makedirs(log_dir, exist_ok=True)

    rank = dist.get_rank() if dist.is_initialized() else 0

    # 清除所有现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - RANK {rank} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_dice_score(logits, targets, smooth=1e-6):
    """计算Dice系数 metric。"""
    # 确保targets是float类型
    targets = targets.float()

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def visualize_predictions(epoch, output_dir, save_dir, images, targets, preds, val_loader):
    """可视化验证集上的预测结果并保存为图片。"""
    # 反归一化
    # 确保 dataset 对象存在 mean 和 std 属性
    if not hasattr(val_loader.dataset, 'mean') or not hasattr(val_loader.dataset, 'std'):
         logging.warning("Dataset object missing 'mean' or 'std' attribute. Skipping denormalization.")
         mean, std = 0.0, 1.0 # 使用默认值避免错误
    else:
        mean, std = val_loader.dataset.mean, val_loader.dataset.std

    # 将 mean 和 std 转换为 tensor 以进行广播
    num_channels = images.shape[1]
    if isinstance(mean, (list, tuple)):
        mean_tensor = torch.tensor(mean).view(1, num_channels, 1, 1)
    else: # 假设是标量
        mean_tensor = torch.tensor([mean] * num_channels).view(1, num_channels, 1, 1)

    if isinstance(std, (list, tuple)):
        std_tensor = torch.tensor(std).view(1, num_channels, 1, 1)
    else: # 假设是标量
        std_tensor = torch.tensor([std] * num_channels).view(1, num_channels, 1, 1)

    mean_tensor = mean_tensor.to(images.device)
    std_tensor = std_tensor.to(images.device)


    fig, axes = plt.subplots(3, 7, figsize=(28, 12))
    fig.suptitle(f'Epoch {epoch + 1} Validation Results (UNETR)', fontsize=20) # 更新标题

    num_samples_to_show = min(7, images.shape[0])

    for i in range(num_samples_to_show):
        img_tensor = images[i]
        # 反归一化
        img_denorm = img_tensor * std_tensor.squeeze() + mean_tensor.squeeze()
        img_denorm = img_denorm.clamp(0, 1)

        # 转换为numpy用于显示
        img_np = img_denorm.cpu().numpy()
        if img_np.shape[0] == 1: # 单通道
            img_np = img_np.squeeze(0)
            cmap = 'gray'
        else: # 多通道
            img_np = img_np.transpose(1, 2, 0)
            cmap = None

        tgt = targets[i].cpu().numpy().squeeze()
        prd = preds[i].cpu().numpy().squeeze()

        axes[0, i].imshow(img_np, cmap=cmap)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(tgt, cmap='gray')
        axes[1, i].set_title(f'Ground Truth')
        axes[1, i].axis('off')

        axes[2, i].imshow(prd, cmap='gray')
        axes[2, i].set_title(f'Prediction')
        axes[2, i].axis('off')

    # 隐藏多余的子图
    for i in range(num_samples_to_show, 7):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, save_dir, f"validation_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    # 确保日志记录只在主进程发生
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if is_main_process:
        logging.info(f"验证结果可视化已保存至: {save_path}")


def evaluate(model, val_loader, criterion, device, epoch, args):
    """评估模型并可视化结果。"""
    model.eval()
    total_val_loss = 0.0
    total_dice_score = 0.0

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    # 用于可视化的容器 (仅在主进程收集)
    all_images, all_targets, all_preds = [], [], []

    with torch.no_grad():
        # 添加 tqdm 进度条，仅在主进程显示
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(not is_main_process))
        for images, targets in val_iterator:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # 确保targets是float类型
            targets = targets.float()

            # 使用bfloat16进行推理
            with autocast(dtype=torch.bfloat16):
                logits = model(images)
                # 使用全精度计算损失
                loss = criterion(logits.float(), targets)

            # 收集损失和指标
            batch_loss = loss.item()
            batch_dice = get_dice_score(logits, targets).item()

            # --- DDP 同步 Batch 指标 ---
            if dist.is_initialized():
                loss_tensor = torch.tensor(batch_loss, device=device)
                dice_tensor = torch.tensor(batch_dice, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(dice_tensor, op=dist.ReduceOp.AVG)
                batch_loss = loss_tensor.item()
                batch_dice = dice_tensor.item()
            # --- DDP 同步结束 ---

            total_val_loss += batch_loss
            total_dice_score += batch_dice


            # 在主进程收集数据用于可视化
            if is_main_process:
                all_images.append(images.cpu())
                all_targets.append(targets.cpu())
                all_preds.append((torch.sigmoid(logits).cpu() > 0.5).float())

                # 更新tqdm的后缀信息
                val_iterator.set_postfix({
                     'val_loss': f"{batch_loss:.4f}",
                     'val_dice': f"{batch_dice:.4f}"
                 })


    # 计算整个验证集的平均损失和指标
    num_batches = len(val_loader)
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_score = total_dice_score / num_batches if num_batches > 0 else 0.0

    synced_loss = avg_val_loss
    synced_dice = avg_dice_score

    # 主进程执行可视化
    if is_main_process:
        if all_images and all_targets and all_preds:
            num_val_samples = sum(len(b) for b in all_targets)
            images_to_show = torch.cat(all_images)[:7]
            targets_to_show = torch.cat(all_targets)[:7]
            preds_to_show = torch.cat(all_preds)[:7]

            if images_to_show.shape[0] > 0: # 确保至少有一个样本
                 visualize_predictions(
                    epoch, args.output, args.savefile,
                    images_to_show, targets_to_show, preds_to_show,
                    val_loader
                 )
                 if images_to_show.shape[0] < 7:
                      logging.warning(f"验证集样本数 ({images_to_show.shape[0]}) 少于7，可视化将只显示 {images_to_show.shape[0]} 个样本。")
            else:
                 logging.warning("验证集为空或未能收集到样本，跳过可视化。")
        else:
             logging.warning("未能收集到用于可视化的验证数据。")


    return synced_loss, synced_dice


def train(config, args):
    """主训练函数"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if use_ddp:
        dist.init_process_group(backend='nccl')

    is_main_process = (local_rank == 0)
    if is_main_process:
        setup_logging(args.output, args.savefile)
        logging.info(f"开始 UNETR 训练，使用 {world_size} 个进程，设备类型: {device.type}") # 更新日志信息
        logging.info(f"配置: {config}")
        logging.info(f"参数: {args}")


    # --- 数据加载 ---
    if is_main_process: logging.info("正在加载数据集...")
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        img_size=config['model']['img_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        use_ddp=use_ddp
    )
    # 广播 mean/std 并附加到 val_loader
    if hasattr(train_loader.dataset, 'mean') and hasattr(train_loader.dataset, 'std'):
         mean_std = [train_loader.dataset.mean, train_loader.dataset.std]
         if use_ddp:
             dist.broadcast_object_list(mean_std, src=0)
         val_loader.dataset.mean, val_loader.dataset.std = mean_std[0], mean_std[1]
         if is_main_process:
              logging.info(f"广播的 Mean: {mean_std[0]}, Std: {mean_std[1]}")
    else:
         if is_main_process: logging.warning("Train dataset missing mean/std attributes.")
         val_loader.dataset.mean, val_loader.dataset.std = 0.0, 1.0


    # --- 模型创建 ---
    # !! 修改: 创建 UNETR 模型 !!
    if is_main_process: logging.info(f"正在创建模型: UNETR with {config['model']['backbone_name']} backbone")
    model = create_unetr_model(
        backbone_name=config['model']['backbone_name'],
        img_size=config['model']['img_size'],
        in_chans=config['model']['in_chans'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        feature_indices=tuple(config['model']['feature_indices']), # 从列表转为元组
        decoder_channels=tuple(config['model']['decoder_channels']) # 从列表转为元组
    ).to(device)

    # 可选的模型编译
    if config['training'].get('use_compile', False):
        if is_main_process: logging.info("正在应用 torch.compile()...")
        model = torch.compile(model, mode="max-autotune")

    # 应用分布式数据并行
    if use_ddp:
        # UNETR可能有未使用的参数，特别是在ViT骨干网络部分，如果只提取部分层特征
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if is_main_process:
            logging.info("已启用 DDP，并设置 find_unused_parameters=True。")

    # --- 损失，优化器，调度器 ---
    criterion = DiceBCELoss()
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

    steps_per_epoch_exact = len(train_loader) / accumulation_steps
    optimizer_steps_per_epoch = max(1, int(steps_per_epoch_exact))
    if is_main_process and optimizer_steps_per_epoch != steps_per_epoch_exact:
        logging.warning(f"Train loader length ({len(train_loader)}) / accumulation ({accumulation_steps}) not integer. Steps/epoch: {optimizer_steps_per_epoch}")

    num_training_steps = num_epochs * optimizer_steps_per_epoch
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch

    if is_main_process:
        logging.info(f"总 Epochs: {num_epochs}, 每个 Epoch 优化器步数: {optimizer_steps_per_epoch}")
        logging.info(f"总步数: {num_training_steps}, Warmup 步数: {num_warmup_steps}")

    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps - num_warmup_steps), eta_min=config['training']['eta_min'])
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps), eta_min=config['training']['eta_min'])

    scaler = GradScaler(enabled=True)

    # --- 检查点加载 ---
    start_epoch = 0
    best_dice = 0.0
    checkpoint_dir = os.path.join(args.output, args.savefile)
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

    if args.reload and os.path.exists(latest_checkpoint_path):
        if is_main_process: logging.info(f"从检查点恢复训练: {latest_checkpoint_path}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)
            model_to_load = model.module if use_ddp else model
            missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if is_main_process:
                if missing_keys: logging.warning(f"加载模型权重时缺失键: {missing_keys}")
                if unexpected_keys: logging.warning(f"加载模型权重时意外键: {unexpected_keys}")

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
            else:
                 if is_main_process: logging.warning("检查点中未找到 scaler 状态。")

            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0)
            if is_main_process: logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。最佳 Dice: {best_dice:.4f}")
        except Exception as e:
            if is_main_process: logging.error(f"加载检查点失败: {e}。从头开始训练。")
            start_epoch = 0; best_dice = 0.0


    # --- 训练循环 ---
    if is_main_process:
        logging.info(f"开始训练，从 Epoch {start_epoch + 1} 到 {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", disable=(not is_main_process))

        for i, (images, targets) in enumerate(train_iterator):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            targets = targets.float()

            is_sync_step = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)
            sync_context = contextlib.nullcontext() if (not use_ddp or is_sync_step) else model.no_sync()

            with sync_context:
                with autocast(dtype=torch.bfloat16):
                    logits = model(images)
                    loss = criterion(logits.float(), targets)
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

            if is_sync_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            batch_loss = loss.item() * accumulation_steps
            running_loss += batch_loss

            if is_main_process:
                 train_iterator.set_postfix({
                     'loss': f"{batch_loss:.4f}",
                     'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                 })


        # --- 同步并计算平均训练损失 ---
        if use_ddp:
            total_loss_tensor = torch.tensor(running_loss, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_samples = torch.tensor(len(train_loader.dataset), device=device)
            avg_train_loss = total_loss_tensor.item() / total_samples.item() if total_samples.item() > 0 else 0.0
        else:
            avg_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0


        # --- 评估 ---
        if use_ddp: dist.barrier()
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, epoch, args)

        # --- 日志与检查点 ---
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # 保存最新检查点
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': (model.module if use_ddp else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dice': best_dice,
                'config': config,
                'args': vars(args)
            }
            try:
                torch.save(checkpoint_data, latest_checkpoint_path)
                logging.debug(f"最新检查点已保存: {latest_checkpoint_path}")
            except Exception as e:
                logging.error(f"保存最新检查点失败: {e}")

            # 保存最佳模型
            if val_dice > best_dice:
                best_dice = val_dice
                logging.info(f"新最佳模型，Dice: {best_dice:.4f}. 保存中...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                try:
                    torch.save((model.module if use_ddp else model).state_dict(), best_checkpoint_path)
                except Exception as e:
                     logging.error(f"保存最佳模型失败: {e}")

    # --- 训练结束 ---
    if is_main_process:
        logging.info(f'训练完成. 最佳验证 Dice: {best_dice:.4f}')

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNETR-2D Segmentation Training Script")
    parser.add_argument('--config', type=str, default='./configs/unetr_HDG.yaml', help='配置文件路径') # 默认使用unetr配置
    parser.add_argument('--data_dir', type=str, default=None, help='数据集目录路径 (覆盖配置文件)')
    parser.add_argument('--output', type=str, default=None, help='输出根目录 (覆盖配置文件)')
    parser.add_argument('--savefile', type=str, default=None, help='本次运行的保存文件夹名 (覆盖配置文件)')
    parser.add_argument('--reload', action='store_true', help='从最新的检查点恢复训练')

    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        sys.exit(1)
    except Exception as e:
        print(f"加载配置文件 '{args.config}' 出错: {e}")
        sys.exit(1)

    # 覆盖配置
    if args.data_dir: config.setdefault('data', {})['data_dir'] = args.data_dir
    if args.output: config.setdefault('output', {})['base_dir'] = args.output
    if args.savefile: config.setdefault('output', {})['save_dir'] = args.savefile
    else:
        config.setdefault('output', {})
        config['output']['save_dir'] = config['output'].get('save_dir', config.get('task_name', 'unetr_run'))

    # 设置args
    args.data_dir = config.get('data', {}).get('data_dir', './data')
    args.output = config.get('output', {}).get('base_dir', './output')
    args.savefile = config.get('output', {}).get('save_dir', 'unetr_run')

    # 检查必要配置
    if 'model' not in config or 'img_size' not in config['model']:
        print("错误: 配置文件缺少 'model' 部分或 'model.img_size'。")
        sys.exit(1)

    # 运行训练
    try:
        train(config, args)
    except Exception as e:
         rank = int(os.environ.get("LOCAL_RANK", 0))
         if rank == 0:
             logging.error(f"训练出错: {e}", exc_info=True)
         if dist.is_initialized():
             dist.destroy_process_group()
         sys.exit(1)
