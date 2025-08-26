import os
import sys
import yaml
import logging
import argparse
import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, MultiStepLR
from matplotlib import pyplot as plt
import numpy as np

sys.path.append("./")
from model.mae import MAE
from dataset.imagenet import build_mae_dataloaders
from dataset.utlis import param_groups_lrd # Still useful for LRD

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

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('HOSTNAME', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', "29500")
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def visualize_and_save(original_img, mask, recon_patches, patch_size, loss, step, output_dir, prefix="train"):
    """
    Creates and saves a visualization of the original, masked, and reconstructed images using Matplotlib.
    Includes the crucial denormalization step for accurate visualization.
    """
    # --- 1. 准备张量 ---
    # 将张量分离、移动到 CPU，并转换为 NumPy 数组
    original_img = original_img.cpu().to(torch.float32).permute(1, 2, 0).numpy()
    recon_patches = recon_patches.cpu().to(torch.float32).numpy()
    mask = mask.cpu().numpy()
    
    H, W, C = original_img.shape
    N = mask.shape[0]
    num_patches_w = W // patch_size
    
    # --- 2. 创建被遮蔽的图像 ---
    masked_img = original_img.copy()
    for i in range(N):
        if mask[i]:  # 检查 True (被遮蔽) 的图像块
            h_idx = i // num_patches_w
            w_idx = i % num_patches_w
            start_h, start_w = h_idx * patch_size, w_idx * patch_size
            masked_img[start_h:start_h + patch_size, start_w:start_w + patch_size, :] = 0 # 涂黑

    # --- 3. 创建重建的图像 (添加了反归一化) ---
    reconstructed_img = original_img.copy()
    
    # --- 关键修改：修正 Reshape 的逻辑 ---
    # 1. 首先按照模型的平面输出格式 (C, P, P) 来重塑
    recon_patches_reshaped = recon_patches.reshape(N, C, patch_size, patch_size)
    
    for i in range(N):
        if mask[i]:  # 只处理被遮蔽的图像块
            h_idx = i // num_patches_w
            w_idx = i % num_patches_w
            start_h, start_w = h_idx * patch_size, w_idx * patch_size
            
            # 2. 将单个图像块从 (C, P, P) 转置为 (P, P, C) 以便后续处理
            recon_patch_chw = recon_patches_reshaped[i]
            recon_patch_hwc = recon_patch_chw.transpose(1, 2, 0)
            
            # --- 反归一化 ---
            original_patch = original_img[start_h:start_h + patch_size, start_w:start_w + patch_size, :]
            mean = original_patch.mean(axis=(0, 1))
            std = original_patch.std(axis=(0, 1))
            
            denorm_patch = recon_patch_hwc * (std + 1e-6) + mean
            
            # 将反归一化后的图像块放回图片中
            reconstructed_img[start_h:start_h + patch_size, start_w:start_w + patch_size, :] = np.clip(denorm_patch, 0, 1)
            
    # --- 4. 绘制并保存图像 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Loss: {loss:.4f} | Step: {step}", fontsize=16)

    # 绘制原始图像
    axes[0].imshow(original_img) # imshow 默认处理 [0,1] 范围的浮点数
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 绘制被遮蔽的图像
    axes[1].imshow(masked_img)
    axes[1].set_title("Masked Input")
    axes[1].axis('off')

    # 绘制重建的图像
    axes[2].imshow(reconstructed_img)
    axes[2].set_title("Reconstructed")
    axes[2].axis('off')

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}_{step}.png"))
    plt.close(fig) # 关闭图像以释放内存

# from PIL import Image, ImageDraw, ImageFont

# def visualize_and_save(original_img, mask, recon_patches, patch_size, loss, step, output_dir, prefix="train"):
#     """
#     Creates and saves a visualization of the original, masked, and reconstructed images using PIL.
#     """
#     # --- 1. Prepare Tensors ---
#     # Detach tensors, move to CPU, and convert to NumPy arrays
#     original_img_np = original_img.cpu().to(torch.float32).permute(1, 2, 0).numpy()
#     recon_patches_np = recon_patches.cpu().to(torch.float32).numpy()
#     mask_np = mask.cpu().numpy()

#     # Scale to 0-255 for image creation
#     original_img_np = np.clip(original_img_np * 255, 0, 255).astype(np.uint8)
    
#     H, W, C = original_img_np.shape
#     N = mask_np.shape[0]
#     num_patches_w = W // patch_size

#     # --- 2. Create Individual PIL Images ---

#     # Original Image
#     original_pil = Image.fromarray(original_img_np)

#     # Masked Image
#     masked_np = original_img_np.copy()
#     for i in range(N):
#         if mask_np[i]:  # If it's a masked patch
#             h_idx = i // num_patches_w
#             w_idx = i % num_patches_w
#             start_h, start_w = h_idx * patch_size, w_idx * patch_size
#             masked_np[start_h:start_h + patch_size, start_w:start_w + patch_size, :] = 0 # Black color
#     masked_pil = Image.fromarray(masked_np)

#     # Reconstructed Image
#     # First, denormalize the reconstructed patches if they were normalized during loss calculation
#     # NOTE: This step assumes the model output is normalized. If your model outputs pixel values
#     # in the [0, 1] range, this denormalization is crucial. We'll get the mean and std
#     # from the original image patches for a more accurate visualization.
    
#     reconstructed_np = original_img_np.copy()
#     recon_patches_reshaped = recon_patches_np.reshape(N, patch_size, patch_size, C)

#     for i in range(N):
#         if mask_np[i]: # If it's a masked patch
#             h_idx = i // num_patches_w
#             w_idx = i % num_patches_w
#             start_h, start_w = h_idx * patch_size, w_idx * patch_size
            
#             # Get the original patch to find its mean and std for denormalization
#             original_patch = original_img_np[start_h:start_h + patch_size, start_w:start_w + patch_size, :]
#             mean = original_patch.mean(axis=(0, 1))
#             std = original_patch.std(axis=(0, 1))
            
#             # Denormalize the reconstructed patch
#             recon_patch = recon_patches_reshaped[i]
#             denorm_patch = recon_patch * (std + 1e-6) + mean
            
#             # Scale to 0-255 and place in the image
#             recon_patch_uint8 = np.clip(denorm_patch, 0, 255).astype(np.uint8)
#             reconstructed_np[start_h:start_h + patch_size, start_w:start_w + patch_size, :] = recon_patch_uint8
            
#     reconstructed_pil = Image.fromarray(reconstructed_np)

#     # --- 3. Combine Images and Add Title ---
    
#     # Define layout properties
#     padding = 10
#     title_height = 40
#     total_width = (W * 3) + (padding * 2)
#     total_height = H + title_height + padding

#     # Create a new blank image with a white background
#     dst = Image.new('RGB', (total_width, total_height), color='white')

#     # Paste the three images side-by-side
#     dst.paste(original_pil, (0, title_height))
#     dst.paste(masked_pil, (W + padding, title_height))
#     dst.paste(reconstructed_pil, (W * 2 + padding * 2, title_height))

#     # Add text
#     draw = ImageDraw.Draw(dst)
#     try:
#         # Use a truetype font if available
#         font = ImageFont.truetype("arial.ttf", 20)
#     except IOError:
#         # Otherwise, use the default font
#         font = ImageFont.load_default()
        
#     title_text = f"Loss: {loss:.4f} | Step: {step}"
#     draw.text((padding, padding // 2), title_text, fill='black', font=font)

#     # --- 4. Save the Final Image ---
#     os.makedirs(output_dir, exist_ok=True)
#     dst.save(os.path.join(output_dir, f"{prefix}_{step}.png"))

def pretrain_mae_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device_id, args, start_epoch, is_ddp=False):
    """
    MAE pre-training loop.
    """
    scaler = GradScaler(enabled=True)
    num_epochs = config['training']['num_epochs']
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)

    if is_main_process:
        logging.info("Starting MAE pre-training for %d epochs with AMP...", num_epochs)
        logging.info("开始BF16优化训练...")
        logging.info(f"torch.compile: {config['training'].get('use_compile', False)}, Fused Optimizer: {config['training'].get('use_fused_optimizer', False)}")
        logging.info(f"将从 Epoch {start_epoch + 1} 开始训练...")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        if is_ddp: train_loader.sampler.set_epoch(epoch)
        model.train()
        
        running_loss = 0.0
        
        for i, (images, _) in enumerate(train_loader):
            # images = images.to(device_id, non_blocking=True)
            
            sync_context = model.no_sync() if (is_ddp and (i + 1) % accumulation_steps != 0) else contextlib.nullcontext()
            
            with sync_context:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    images = images.to(device_id, non_blocking=True)
                    # MAE model's forward pass returns the loss directly
                    loss, recon_patches_flat, mask = model(images)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler and not config['training']['use_postrain']: scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            # Save images every 1000 steps on the main process
            if (i + 1) % 500 == 0 and is_main_process:
                with torch.no_grad():
                    # MAE model returns un-normalized patches
                    loss_val, recon, mask_val = model(images)
                    visualize_and_save(
                        images[0], 
                        mask_val[0], 
                        recon[0], 
                        args.patch_size, 
                        loss_val.item(), 
                        i + 1,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"train_e{epoch + 1}"
                    )

            if (i + 1) % 10 == 0 and is_main_process:
                avg_loss = running_loss / 10
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'[Epoch {epoch + 1}, Batch {i + 1}] Train Loss: {avg_loss:.5f}, current_lr: {current_lr:.6f}')
                running_loss = 0.0

        # Evaluate on the validation set
        val_loss = evaluate_mae_model(
            model,
            val_loader,
            device_id,
            args,
            is_ddp=is_ddp,
            epoch=epoch
        )
                
        if is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch {epoch + 1}/{num_epochs} | Val Loss: {val_loss:.5f} | current_lr: {current_lr:.6f}")
            
            # Save checkpoints
            checkpoint_dir = os.path.join(args.output, args.savefile)
            latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, latest_checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.info(f"新的最佳验证损失: {best_val_loss:.5f}. 保存最佳模型...")
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, best_checkpoint_path)

    if is_main_process:
        logging.info(f'Finished Pre-training. Best Validation Loss: {best_val_loss:.5f}')


def evaluate_mae_model(model, val_loader, device, args, is_ddp=False, epoch=0):
    """
    Evaluation function for MAE pre-training.
    Returns the average reconstruction loss on the validation set.
    """
    model.eval()
    total_loss, total_samples = 0, 0
    is_main_process = not is_ddp or (is_ddp and dist.get_rank() == 0)
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                loss, recon, mask = model(images)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                # Save images on the main process for the first 10 validation batches
                if i < 10 and is_main_process:
                     visualize_and_save(
                        images[0], 
                        mask[0], 
                        recon[0], 
                        args.patch_size, 
                        loss.item(), 
                        i,
                        os.path.join(args.output, args.savefile, "images"),
                        prefix=f"val_e{epoch + 1}"
                    )

    if is_ddp:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        dist.all_reduce(total_loss_tensor)
        dist.all_reduce(total_samples_tensor)
        total_loss, total_samples = total_loss_tensor.item(), total_samples_tensor.item()
        
    return total_loss / total_samples if total_samples > 0 else 0

def mae_imagenet_pretrain_single(args, config):
    """Main DDP pre-training function for MAE."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = 'nccl'
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

    if local_rank == 0:
        setup_logging(args)
        logging.info(f"开始预训练，使用 {world_size} 个进程，设备类型: {device.type}")

    args.img_size = config['model']['img_size']
    args.batch_size = config['training']['batch_size']
    args.patch_size = config['model']['patch_size'] # Get patch size for visualization
    
    dataloaders = build_mae_dataloaders(
        img_size=args.img_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # 实例化 MAE 模型
    model = MAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_dim=config['model']['encoder_embed_dim'],
        encoder_depth=config['model']['encoder_depth'],
        encoder_heads=config['model']['encoder_heads'],
        decoder_dim=config['model']['decoder_embed_dim'],
        decoder_depth=config['model']['decoder_depth'],
        decoder_heads=config['model']['decoder_heads'],
        mask_ratio=config['model']['mask_ratio']
    ).to(device)

    if config['training'].get('use_compile', False):
        if dist.get_rank() == 0: logging.info("正在应用 torch.compile()...")
        model = torch.compile(model)
        
    if device.type == 'cuda' or world_size > 1:
        if device.type == 'cuda':
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)
    
    use_fused = config['training'].get('use_fused_optimizer', False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay'],
        betas=tuple(config['training'].get('betas', (0.9, 0.95))),
        fused=use_fused,
        amsgrad=False
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
    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(args.output, args.savefile)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

    if args.reload and os.path.exists(checkpoint_path):
        if dist.get_rank() == 0:
            logging.info(f"从检查点恢复预训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if dist.get_rank() == 0:
            logging.info(f"成功恢复，将从 Epoch {start_epoch + 1} 开始。")
            
    pretrain_mae_model(model, dataloaders['train'], dataloaders['val'], optimizer, scheduler, config['training']['num_epochs'], device, args, start_epoch=start_epoch, is_ddp=(world_size > 1))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE Pre-training Script")
    
    parser.add_argument('--config', type=str, default='./configs/mae-vit-b16_IN1K.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--task', type=str, default='mae_pretrain', help='Type of task')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory')
    parser.add_argument('--savefile', type=str, default='mae_vit-b16', help='Subdirectory for saving logs and models')
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
    
    mae_imagenet_pretrain_single(args, config)