import os
import sys
# 确保可以找到自定义模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..')) # 假设脚本在项目根目录的子文件夹中
if project_root not in sys.path:
    sys.path.append(project_root)

import yaml
import logging
import argparse
import contextlib
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast


from model.sam import create_sam_b_variant
from model.losses import DiceBCELoss
from dataset.hdg import create_dataloaders



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
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(images.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(images.device)

    fig, axes = plt.subplots(3, 7, figsize=(28, 12))
    fig.suptitle(f'Epoch {epoch + 1} Validation Results', fontsize=20)

    num_samples_to_show = min(7, images.shape[0]) # 处理最后一个batch可能小于7的情况

    for i in range(num_samples_to_show):
        img_tensor = images[i]
        # 反归一化
        img_denorm = img_tensor * std.squeeze() + mean.squeeze()
        img_denorm = img_denorm.clamp(0, 1)

        # 转换为numpy用于显示 (C, H, W) -> (H, W, C) or (H, W)
        img_np = img_denorm.cpu().numpy()
        if img_np.shape[0] == 1: # 单通道
            img_np = img_np.squeeze(0)
            cmap = 'gray'
        else: # 多通道 (e.g., RGB)
            img_np = img_np.transpose(1, 2, 0)
            cmap = None # imshow会自动处理RGB

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

    # 隐藏多余的子图（如果验证集样本少于7）
    for i in range(num_samples_to_show, 7):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, save_dir, f"validation_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    if dist.get_rank() == 0: # 仅主进程记录日志
        logging.info(f"验证结果可视化已保存至: {save_path}")


def evaluate(model, val_loader, criterion, device, epoch, args):
    """评估模型并可视化结果。"""
    model.eval()
    total_val_loss = 0.0
    total_dice_score = 0.0

    # 用于可视化的容器 (仅在主进程收集)
    all_images, all_targets, all_preds = [], [], []

    with torch.no_grad():
        # 添加 tqdm 进度条，仅在主进程显示
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(dist.get_rank() != 0))
        for images, targets in val_iterator:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # 确保targets是float类型
            targets = targets.float()

            # 使用bfloat16进行推理
            with autocast(dtype=torch.bfloat16):
                logits = model(images)
                # 使用全精度计算损失以获得更精确的值
                loss = criterion(logits.float(), targets) # 将logits转回float32计算损失

            # 收集损失和指标
            batch_loss = loss.item()
            batch_dice = get_dice_score(logits, targets).item()
            total_val_loss += batch_loss
            total_dice_score += batch_dice

            # 在主进程收集数据用于可视化
            if dist.get_rank() == 0:
                # 只收集第一个batch的图像用于可视化，以节省内存
                # 或者收集所有？需要确认内存占用
                # 收集所有样本以匹配可视化代码
                all_images.append(images.cpu()) # 移动到CPU以节省GPU内存
                all_targets.append(targets.cpu())
                all_preds.append((torch.sigmoid(logits).cpu() > 0.5).float())

            # 更新tqdm的后缀信息
            if dist.get_rank() == 0:
                 val_iterator.set_postfix({
                     'val_loss': f"{batch_loss:.4f}",
                     'val_dice': f"{batch_dice:.4f}"
                 })


    # 计算整个验证集的平均损失和指标
    num_batches = len(val_loader)
    avg_val_loss = total_val_loss / num_batches
    avg_dice_score = total_dice_score / num_batches

    # 在DDP环境中同步所有进程的平均评估结果
    metrics_tensor = torch.tensor([avg_val_loss, avg_dice_score], device=device)
    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)

    synced_loss, synced_dice = metrics_tensor.tolist()

    # 主进程执行可视化
    if dist.get_rank() == 0:
        if all_images and all_targets and all_preds:
            # 确保只可视化最多7个样本
            num_val_samples = sum(len(b) for b in all_targets)
            images_to_show = torch.cat(all_images)[:7]
            targets_to_show = torch.cat(all_targets)[:7]
            preds_to_show = torch.cat(all_preds)[:7]

            if num_val_samples >= 7:
                 visualize_predictions(
                    epoch, args.output, args.savefile,
                    images_to_show, targets_to_show, preds_to_show,
                    val_loader
                 )
            else:
                 logging.warning(f"验证集样本数 ({num_val_samples}) 少于7，无法生成完整的可视化图。")
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
        logging.info(f"开始训练，使用 {world_size} 个进程，设备类型: {device.type}")
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
    # 将计算出的mean和std附加到val_loader.dataset以供可视化使用
    # 确保只在主进程执行或所有进程都能访问到
    if hasattr(train_loader.dataset, 'mean') and hasattr(train_loader.dataset, 'std'):
         val_loader.dataset.mean, val_loader.dataset.std = train_loader.dataset.mean, train_loader.dataset.std
    else:
         if is_main_process:
             logging.warning("Train dataset object missing 'mean' or 'std' attribute. Visualization might not denormalize correctly.")
         # 在其他进程上也设置默认值，以避免潜在的属性错误
         val_loader.dataset.mean, val_loader.dataset.std = 0.0, 1.0


    # --- 模型创建 ---
    if is_main_process: logging.info(f"正在创建模型: SAM-B Variant for Segmentation")
    # create_sam_b_variant 会根据 img_size 自动设置 in_chans
    model = create_sam_b_variant(
        img_size=config['model']['img_size'],
        target_patch_grid_size=config['model']['target_patch_grid_size'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        decoder_channels=tuple(config['model']['decoder_channels']) # 从列表转为元组
    ).to(device)

    # 可选的模型编译
    if config['training'].get('use_compile', False):
        if is_main_process: logging.info("正在应用 torch.compile()...")
        # 推荐使用 'max-autotune' 模式以获得最佳性能
        model = torch.compile(model, mode="max-autotune")

    # 应用分布式数据并行
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # find_unused_parameters=False 通常更高效

    # --- 损失，优化器，调度器 ---
    criterion = DiceBCELoss()
    # 过滤掉不需要计算梯度的参数（例如，冻结的骨干网络层，如果有的话）
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

    # 确保 optimizer_steps_per_epoch > 0
    steps_per_epoch_exact = len(train_loader) / accumulation_steps
    optimizer_steps_per_epoch = max(1, int(steps_per_epoch_exact)) # 至少为1
    if is_main_process and optimizer_steps_per_epoch != steps_per_epoch_exact:
        logging.warning(f"Train loader length ({len(train_loader)}) is not perfectly divisible by accumulation steps ({accumulation_steps}). Effective steps per epoch: {optimizer_steps_per_epoch}")


    num_training_steps = num_epochs * optimizer_steps_per_epoch
    num_warmup_steps = warmup_epochs * optimizer_steps_per_epoch

    if is_main_process:
        logging.info(f"总训练 Epochs: {num_epochs}")
        logging.info(f"每个 Epoch 的优化器步数: {optimizer_steps_per_epoch}")
        logging.info(f"总优化器步数: {num_training_steps}")
        logging.info(f"Warmup 步数: {num_warmup_steps}")


    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps - num_warmup_steps), eta_min=config['training']['eta_min']) # T_max 至少为1
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[num_warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_training_steps), eta_min=config['training']['eta_min']) # T_max 至少为1

    # 初始化混合精度训练的 GradScaler
    scaler = GradScaler(enabled=True) # 总是启用，autocast会处理bf16

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
            # 区分DDP模型和非DDP模型加载
            model_to_load = model.module if use_ddp else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('best_dice', 0.0) # 使用get以兼容旧的checkpoint
            if is_main_process: logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。最佳 Dice: {best_dice:.4f}")
        except Exception as e:
            if is_main_process:
                logging.error(f"加载检查点失败: {e}。将从头开始训练。")
            start_epoch = 0
            best_dice = 0.0


    # --- 训练循环 ---
    if is_main_process:
        logging.info(f"开始训练，从 Epoch {start_epoch + 1} 到 {num_epochs}...")


    for epoch in range(start_epoch, num_epochs):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch) # 设置 DDP sampler 的 epoch
        model.train() # 设置模型为训练模式

        running_loss = 0.0
        # 使用tqdm创建训练进度条，仅在主进程显示
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", disable=(not is_main_process))

        # 在每个epoch开始时清除梯度
        # optimizer.zero_grad(set_to_none=True) # 移动到梯度累积步骤后

        for i, (images, targets) in enumerate(train_iterator):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # 确保targets是float类型
            targets = targets.float()

            # 决定是否需要同步梯度 (DDP)
            is_sync_step = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)
            sync_context = contextlib.nullcontext() if is_sync_step else model.no_sync()

            with sync_context:
                # 使用bfloat16进行前向传播
                with autocast(dtype=torch.bfloat16):
                    logits = model(images)
                    # 使用全精度计算损失
                    loss = criterion(logits.float(), targets) # 将logits转回float32计算损失
                    loss = loss / accumulation_steps # 梯度累积标准化

                # 使用 GradScaler 进行反向传播
                scaler.scale(loss).backward()

            # 仅在同步步骤执行优化器更新和学习率调度
            if is_sync_step:
                # 可以选择梯度裁剪 (如果需要)
                # scaler.unscale_(optimizer) # 如果需要 unscale
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # 清除梯度
                scheduler.step() # 更新学习率

            batch_loss = loss.item() * accumulation_steps # 获取未标准化的损失值
            running_loss += batch_loss

            # 更新tqdm的后缀信息
            if is_main_process:
                 train_iterator.set_postfix({
                     'loss': f"{batch_loss:.4f}",
                     'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                 })


        # --- 在所有进程之间同步并计算平均训练损失 ---
        if use_ddp:
            total_loss_tensor = torch.tensor(running_loss, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            # 总损失 / (总样本数) = 平均损失
            # 注意：len(train_loader.dataset) 是总样本数
            avg_train_loss = total_loss_tensor.item() / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0
        else:
            # 单进程情况
            avg_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0.0


        # --- 评估 ---
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, epoch, args)

        # --- 日志记录与检查点保存 (仅在主进程) ---
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
                'config': config, # 保存配置以供参考
                'args': args # 保存命令行参数
            }
            torch.save(checkpoint_data, latest_checkpoint_path)
            logging.debug(f"最新检查点已保存至: {latest_checkpoint_path}")

            # 如果是最佳模型，则另外保存一份（只保存模型权重以节省空间）
            if val_dice > best_dice:
                best_dice = val_dice
                logging.info(f"发现新的最佳模型，Dice: {best_dice:.4f}. 正在保存...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save((model.module if use_ddp else model).state_dict(), best_checkpoint_path)

    # --- 训练结束 ---
    if is_main_process:
        logging.info(f'训练完成. 最佳验证Dice系数: {best_dice:.4f}')

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-B Segmentation Training Script")
    parser.add_argument('--config', type=str, default='./configs/sam-b_HDG.yaml', help='配置文件路径')
    # 数据和输出路径通常通过配置文件设置，但保留命令行覆盖选项
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
        print(f"错误: 配置文件 '{args.config}' 未找到。请确保路径正确。")
        sys.exit(1)
    except Exception as e:
        print(f"加载配置文件 '{args.config}' 时出错: {e}")
        sys.exit(1)


    # 使用命令行参数覆盖配置文件设置 (如果提供了)
    # 确保嵌套字典的键存在
    if args.data_dir:
        config.setdefault('data', {})['data_dir'] = args.data_dir
    if args.output:
        config.setdefault('output', {})['base_dir'] = args.output
    if args.savefile:
        config.setdefault('output', {})['save_dir'] = args.savefile
    else:
        # 如果命令行未提供savefile，则使用配置文件中的save_dir或task_name
        config.setdefault('output', {})
        config['output']['save_dir'] = config['output'].get('save_dir', config.get('task_name', 'sam_run'))

    # 将最终确定的路径设置回args，以便在日志和函数调用中使用
    args.data_dir = config.get('data', {}).get('data_dir', './data') # 提供默认值
    args.output = config.get('output', {}).get('base_dir', './output') # 提供默认值
    args.savefile = config.get('output', {}).get('save_dir', 'sam_run') # 提供默认值

    # 确保模型配置存在
    if 'model' not in config:
        print("错误: 配置文件中缺少 'model' 部分。")
        sys.exit(1)
    if 'img_size' not in config['model']:
         print("错误: 配置文件 'model' 部分缺少 'img_size'。")
         sys.exit(1)

    # 运行训练
    train(config, args)
