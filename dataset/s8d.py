import os
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist
from tqdm import tqdm
import logging
from typing import List, Optional # For type hinting
import sys # For sys.exit

_logger = logging.getLogger(__name__)

# --- Script to calculate dataset statistics ---

class Spring8DatasetForStats(Dataset):
    """用于计算统计信息的数据集。"""
    def __init__(self, all_files, resolution):
        self.resolution = resolution
        self.image_filenames = all_files
        # 只转换为 [0, 1] 的 float Tensor
        # Removed transforms.ToTensor() as it expects uint8 or PIL
        # Manual scaling is done below

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        try:
            # 直接读取 uint16 数据
            image = tifffile.imread(img_path).astype(np.float32)
            # 手动归一化到 [0, 1]
            image_normalized = image / 65535.0
            # 添加通道维度 C, H, W
            # Check if image is already HWC or HW
            if image_normalized.ndim == 2: # HW
                 image_normalized = np.expand_dims(image_normalized, axis=0) # CHW
            elif image_normalized.ndim == 3 and image_normalized.shape[-1] == 1 : # HWC
                 image_normalized = image_normalized.transpose(2, 0, 1) # CHW
            elif image_normalized.ndim == 3 and image_normalized.shape[0] == 1: # CHW already
                 pass
            else:
                 _logger.warning(f"Unexpected image shape {image.shape} for stats calc at {img_path}. Assuming grayscale.")
                 image_normalized = image_normalized[0:1,:,:] if image_normalized.ndim==3 else np.expand_dims(image_normalized, axis=0)


            # 转为 Tensor C, H, W
            return torch.from_numpy(image_normalized)
        except Exception as e:
            _logger.error(f"Error loading image {img_path} for stats: {e}")
            # Return a tensor of zeros or handle appropriately
            return torch.zeros((1, self.resolution, self.resolution), dtype=torch.float32)


def calculate_stats_s8d(slice_dir, num_workers=4, batch_size=32):
    """计算 s8d 数据集的均值和标准差 (仅在 Rank 0 执行计算)。"""
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    mean = torch.tensor([0.0]) # Default values
    std = torch.tensor([1.0])  # Default values

    if is_main_process:
        _logger.info("Rank 0: 开始计算 s8d 数据集的均值和标准差...")
        manifest_path = os.path.join(slice_dir, 'slice_manifest.csv')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到清单文件: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        # Filter out potential missing files listed in manifest
        all_files = []
        for p in manifest['image_path']:
            full_path = os.path.join(slice_dir, p)
            if os.path.exists(full_path):
                all_files.append(full_path)
            else:
                 _logger.warning(f"Manifest lists file not found, skipping for stats: {full_path}")
        if not all_files:
             raise RuntimeError("No valid image files found based on the manifest for stats calculation.")


        # 推断分辨率
        try:
            sample_img = tifffile.imread(all_files[0])
            resolution = sample_img.shape[0] # Assume square
            _logger.info(f"Rank 0: 推断图像分辨率为: {resolution}x{resolution}")
        except Exception as e:
            raise RuntimeError(f"无法读取示例图像 {all_files[0]} 来推断分辨率: {e}")

        stat_dataset = Spring8DatasetForStats(all_files, resolution)
        stat_loader = DataLoader(stat_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

        num_pixels = 0 # Use num_pixels instead of num_samples
        sum_pixels = torch.zeros(1, dtype=torch.float64) # Use float64 for accumulation
        sum_sq_pixels = torch.zeros(1, dtype=torch.float64)


        for images in tqdm(stat_loader, desc="Rank 0: 计算统计数据"):
            # images shape: B, C, H, W (ensure C=1)
            if images.shape[1] != 1:
                _logger.warning(f"Stats calculation expected 1 channel, got {images.shape[1]}. Using first channel.")
                images = images[:, 0:1, :, :]
            sum_pixels += images.sum().double() # Sum over all dimensions, accumulate in float64
            sum_sq_pixels += (images ** 2).sum().double()
            num_pixels += images.numel() # Total number of pixels in the batch

        if num_pixels > 0:
            mean = (sum_pixels / num_pixels).float() # Convert back to float32
            # Clamping variance calculation for numerical stability
            variance = (sum_sq_pixels / num_pixels - mean.double() ** 2).clamp(min=1e-6)
            std = torch.sqrt(variance).float()
        else:
             _logger.warning("Rank 0: 未能计算统计数据，数据集为空或无法加载？")
             mean = torch.tensor([0.0])
             std = torch.tensor([1.0])


        _logger.info(f"Rank 0: 计算完成。均值: {mean.item():.4f}, 标准差: {std.item():.4f}")

    # 将计算结果广播到所有进程
    if dist.is_initialized():
        # Ensure tensors are created even on non-main ranks for broadcast
        if not is_main_process:
            mean = torch.empty(1, dtype=torch.float32)
            std = torch.empty(1, dtype=torch.float32)

        stats_tensor = torch.cat([mean, std]).cuda(dist.get_rank()) # Move to current device
        dist.broadcast(stats_tensor, src=0)
        mean = stats_tensor[0].cpu() # Move back to CPU
        std = stats_tensor[1].cpu()
        dist.barrier() # Ensure all ranks have received the stats

    # 返回标量值
    return mean.item(), std.item()


class S8DFinetune2D(Dataset):
    """用于加载 s8d 2D 切片的 PyTorch 数据集。"""

    def __init__(self, slice_dir, slice_ids: List[str], num_classes=5, transform=None, target_transform=None, mean=0.0, std=1.0, img_size: Optional[int] = None):
        """ Modified to accept slice_ids directly and optional img_size """
        self.slice_dir = slice_dir
        self.transform = transform # 用于数据增强
        self.num_classes = num_classes
        self.target_transform = target_transform # 用于标签增强
        self.manifest_full = self._load_manifest() # Load full manifest
        self.mean = mean
        self.std = std
        self.img_size = img_size # Store target img_size

        # Filter manifest based on provided slice_ids
        # Ensure slice_id column type matches for isin
        self.manifest_full['slice_id'] = self.manifest_full['slice_id'].astype(str)
        slice_ids_str = [str(s) for s in slice_ids]
        self.manifest = self.manifest_full[self.manifest_full['slice_id'].isin(slice_ids_str)].reset_index(drop=True)
        if len(self.manifest) != len(slice_ids):
             _logger.warning(f"Manifest filtering resulted in {len(self.manifest)} samples, but {len(slice_ids)} were expected. Some slice IDs might be missing in the manifest.")


        # 标准化变换（在数据增强之后应用）
        # Handle potential zero std deviation
        self.normalize = transforms.Normalize(mean=[self.mean], std=[max(self.std, 1e-6)])


    def _load_manifest(self):
        manifest_path = os.path.join(self.slice_dir, 'slice_manifest.csv')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到清单文件: {manifest_path}。")
        try:
            return pd.read_csv(manifest_path)
        except Exception as e:
             raise RuntimeError(f"无法读取清单文件 {manifest_path}: {e}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # Handle potential index out of bounds with DistributedSampler padding
        if idx >= len(self.manifest):
             _logger.warning(f"Index {idx} out of bounds for manifest length {len(self.manifest)}. Returning dummy data.")
             h = w = self.img_size if self.img_size is not None else 512 # Use target or default size
             return torch.zeros((1, h, w), dtype=torch.float32), torch.zeros((h, w), dtype=torch.long), "dummy_slice"

        record = self.manifest.iloc[idx]
        slice_id_val = record['slice_id'] # Get slice_id for error reporting

        # 加载图像和标签
        img_path = os.path.join(self.slice_dir, record['image_path'])
        label_path = os.path.join(self.slice_dir, record['label_path'])
        try:
            # 读取 uint16 图像并转为 float32 [0.0, 1.0]
            img = tifffile.imread(img_path).astype(np.float32) / 65535.0
            label = tifffile.imread(label_path) # 标签通常是 uint8 或 int

            # Add channel dimension if image is HW -> HWC for albumentations
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)

        except FileNotFoundError as e:
            _logger.error(f"Slice {slice_id_val}: 无法加载文件: {e}")
            raise FileNotFoundError(f"无法加载索引 {idx} (slice_id: {slice_id_val}) 的文件: {e}")
        except Exception as e:
            _logger.error(f"Slice {slice_id_val}: 加载索引 {idx} ({img_path}, {label_path}) 时出错: {e}")
            raise RuntimeError(f"加载索引 {idx} (slice_id: {slice_id_val}) 时出错: {e}")

        # --- Resize BEFORE augmentation if img_size is specified ---
        # Using Albumentations for resize to handle image and mask together if needed
        # But here, only resizing image before transform. Mask resizing happens later if needed.
        # This assumes transform pipeline handles resizing internally if provided.
        # Let's handle resize explicitly *before* custom transform if img_size is set.
        current_h, current_w = img.shape[:2]
        target_h = target_w = self.img_size if self.img_size is not None else current_h

        if self.img_size is not None and (current_h != target_h or current_w != target_w):
             try:
                 import cv2 # Use cv2 for resizing numpy arrays
                 img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                 # Add channel back if lost during resize
                 if img.ndim == 2: img = np.expand_dims(img, axis=-1)

                 # Resize label using nearest neighbor interpolation
                 label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

             except ImportError:
                  _logger.error("OpenCV (cv2) not found, required for resizing when img_size is specified. Install with 'pip install opencv-python'. Skipping resize.")
             except Exception as e:
                  _logger.error(f"Slice {slice_id_val}: Resizing failed: {e}")
                  # Decide how to handle: skip sample, return original, raise error?
                  # Let's raise error for now
                  raise RuntimeError(f"Resizing slice {slice_id_val} failed: {e}")


        # 应用数据增强 (Albumentations 或 torchvision)
        # Assumes img is HWC numpy array
        if self.transform:
            try:
                augmented = self.transform(image=img, mask=label) # Apply to both if transform supports it
                img = augmented['image']
                label = augmented['mask']
            except Exception as e:
                 _logger.error(f"Slice {slice_id_val}: 应用图像/标签变换时出错: {e}", exc_info=True)
                 # Fallback or raise error? Fallback to un-augmented for now
                 # Need original img shape again if fallback
                 # Let's re-read img/label or raise error
                 raise RuntimeError(f"Augmentation failed for slice {slice_id_val}: {e}")

        # 标签变换 (already done in transform if applicable)
        # if self.target_transform:
        #      try:
        #          label = self.target_transform(mask=label)['mask']
        #      except Exception as e:
        #           _logger.error(f"Slice {slice_id_val}: 应用标签变换时出错: {e}", exc_info=True)


        # 转换为张量 (CHW for PyTorch)
        if not isinstance(img, torch.Tensor):
             if img.ndim == 3: # HWC -> CHW
                 img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
             elif img.ndim == 2: # HW -> CHW (C=1)
                 img_tensor = torch.from_numpy(img).unsqueeze(0)
             else:
                 raise ValueError(f"Slice {slice_id_val}: Unexpected image dimensions after transform: {img.ndim}")
        else: # Already a tensor
             if img.ndim == 2: img_tensor = img.unsqueeze(0)
             # Ensure CHW if C=1 was squeezed during transform
             elif img.ndim == 3 and img.shape[0] != 1 and img.shape[-1]==1: # HWC?
                 img_tensor = img.permute(2, 0, 1)
             elif img.ndim == 3 and img.shape[0] != 1 and img.shape[0]!=3: # Maybe BHWC? Problematic
                 raise ValueError(f"Slice {slice_id_val}: Unexpected image tensor shape after transform: {img.shape}")
             else: # Assume CHW
                 img_tensor = img


        if not isinstance(label, torch.Tensor):
            label_tensor = torch.from_numpy(label.astype(np.int64)) # HW, LongTensor
        else:
            label_tensor = label.long()


        # 应用标准化
        try:
            img_tensor = self.normalize(img_tensor)
        except Exception as e:
             _logger.error(f"Slice {slice_id_val}: 应用标准化时出错. Img shape: {img_tensor.shape}, Mean: {self.mean}, Std: {self.std}. Error: {e}")
             # Handle error
             raise RuntimeError(f"Normalization failed for slice {slice_id_val}: {e}")


        # Final check for label shape (should be HW)
        if label_tensor.ndim != 2:
            _logger.warning(f"Slice {slice_id_val}: Final label tensor has unexpected shape {label_tensor.shape}. Attempting to squeeze.")
            try:
                label_tensor = label_tensor.squeeze()
                if label_tensor.ndim != 2: raise ValueError("Squeeze failed.")
            except:
                raise ValueError(f"Slice {slice_id_val}: Could not ensure label tensor has 2 dimensions (HW). Final shape: {label_tensor.shape}")


        return img_tensor, label_tensor, slice_id_val # Return original slice_id

    # --- Methods to get volume/slice info (kept from original) ---
    def get_volume_ids_from_manifest(self, manifest_df):
        """ Helper to get unique sorted volume IDs from a manifest dataframe """
        return sorted(manifest_df['volume_id'].unique()) if 'volume_id' in manifest_df else []

    def get_slices_for_volume_from_manifest(self, manifest_df, volume_id):
        """ Helper to get slice IDs for a specific volume ID from a manifest dataframe """
        if 'volume_id' not in manifest_df or 'slice_id' not in manifest_df: return []
        # Ensure type match for comparison
        volume_id_str = str(volume_id)
        manifest_df['volume_id'] = manifest_df['volume_id'].astype(str)
        manifest_df['slice_id'] = manifest_df['slice_id'].astype(str)
        return manifest_df[manifest_df['volume_id'] == volume_id_str]['slice_id'].tolist()


def build_s8d_segmentation_dataloaders(data_dir, batch_size, num_workers, img_size=None, use_ddp=True):
    """
    创建 s8d 数据加载器，自动划分，并应用标准化。
    Splitting logic is performed ONLY ON RANK 0 and broadcasted.
    Args:
        img_size (int, optional): Target size for image resizing. Defaults to None (no resize).
        use_ddp (bool): Whether to set up DistributedSampler.
    """
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # 1. 计算或加载统计数据 (Rank 0 computes, all ranks receive)
    mean, std = calculate_stats_s8d(data_dir, num_workers=num_workers, batch_size=batch_size*2)

    # 2. 划分数据集 (Rank 0 performs split, broadcasts results)
    train_slice_ids = []
    val_slice_ids = []
    # volume_ids = [] # Keep track of volumes for logging # Moved inside rank 0 block

    if is_main_process:
        _logger.info("Rank 0: 开始数据集划分...")
        train_volume_ids, val_volume_ids = [], [] # Init here for logging scope
        try:
            # Load full manifest only on Rank 0
            manifest_full_path = os.path.join(data_dir, 'slice_manifest.csv')
            if not os.path.exists(manifest_full_path):
                 raise FileNotFoundError(f"未找到清单文件: {manifest_full_path}")
            manifest_full = pd.read_csv(manifest_full_path)
            # Create temporary instance to use helper methods
            # Need to provide some dummy slice IDs initially
            all_slice_ids_temp = manifest_full['slice_id'].astype(str).tolist()
            temp_ds = S8DFinetune2D(data_dir, slice_ids=all_slice_ids_temp[:1], mean=mean, std=std, img_size=img_size)

            volume_ids = temp_ds.get_volume_ids_from_manifest(manifest_full)
            if not volume_ids:
                 _logger.warning("未能从清单中获取 volume_ids，将尝试基于所有切片进行随机拆分。")
                 # Fallback: Split slices directly if no volume info
                 all_slice_ids = manifest_full['slice_id'].astype(str).tolist()
                 np.random.seed(42); np.random.shuffle(all_slice_ids)
                 split_idx = int(len(all_slice_ids) * 0.8) # Default 80/20 split
                 train_slice_ids = all_slice_ids[:split_idx]
                 val_slice_ids = all_slice_ids[split_idx:]
            else:
                # Volume-based split (e.g., 80/20)
                np.random.seed(42); np.random.shuffle(volume_ids)
                split_idx = int(len(volume_ids) * 0.8) # Default 80/20 split
                train_volume_ids = volume_ids[:split_idx]
                val_volume_ids = volume_ids[split_idx:]
                _logger.info(f"Rank 0: 训练集 Volume IDs ({len(train_volume_ids)}): {train_volume_ids}")
                _logger.info(f"Rank 0: 验证集 Volume IDs ({len(val_volume_ids)}): {val_volume_ids}")

                # Use helper methods correctly with the full manifest
                train_slice_ids = [s for vol_id in train_volume_ids for s in temp_ds.get_slices_for_volume_from_manifest(manifest_full, vol_id)]
                val_slice_ids = [s for vol_id in val_volume_ids for s in temp_ds.get_slices_for_volume_from_manifest(manifest_full, vol_id)]

            # Ensure lists are not empty
            if not train_slice_ids or not val_slice_ids:
                 raise ValueError("训练或验证切片列表为空。请检查清单文件和拆分逻辑。")

            _logger.info(f"Rank 0: 数据集划分完成: {len(train_slice_ids)} 训练切片, {len(val_slice_ids)} 验证切片。")

        except Exception as e:
            _logger.error(f"Rank 0: 数据集划分失败: {e}", exc_info=True)
            # Signal error to other ranks if DDP is used
            object_list_to_broadcast = [None, None] # Indicate failure
            if use_ddp:
                 # Hacky way to signal error: send None lists
                 try:
                     dist.broadcast_object_list(object_list_to_broadcast, src=0)
                 except Exception as broadcast_e:
                      _logger.error(f"Rank 0: 广播错误信号失败: {broadcast_e}")
            raise e # Re-raise error on rank 0

        # Broadcast success signal if DDP is used
        object_list_to_broadcast = [train_slice_ids, val_slice_ids]


    # Broadcast slice IDs if using DDP
    if use_ddp:
        # Initialize list on non-main ranks for broadcast_object_list
        if not is_main_process:
             object_list_to_broadcast = [None, None]

        dist.broadcast_object_list(object_list_to_broadcast, src=0)
        # Unpack the received lists on non-main processes
        if not is_main_process:
            train_slice_ids = object_list_to_broadcast[0]
            val_slice_ids = object_list_to_broadcast[1]
            # Check if rank 0 failed
            if train_slice_ids is None or val_slice_ids is None:
                 _logger.error(f"Rank {rank}: 从 Rank 0 接收切片 ID 失败。退出。")
                 # Attempt cleanup
                 if dist.is_initialized(): dist.destroy_process_group()
                 sys.exit(1) # Exit non-zero if rank 0 failed

        # Barrier to ensure all ranks have the lists
        dist.barrier()
        # Optional: Log received sizes on other ranks for verification
        # _logger.info(f"Rank {rank}: Received slice IDs. Train: {len(train_slice_ids)}, Val: {len(val_slice_ids)}")

    # Check again if lists are empty after potential broadcast failure handling
    if not train_slice_ids or not val_slice_ids:
        raise ValueError(f"Rank {rank}: 训练或验证切片列表在广播后为空。")

    # 3. 定义数据增强 (示例 - 使用 Albumentations)
    # Import here to avoid making albumentations a hard requirement if not used
    try:
        import albumentations as A
        # Training transforms (applied after potential resize in Dataset)
        train_transform_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Add intensity augmentations if desired
            # A.RandomBrightnessContrast(p=0.2),
        ]
        train_transforms_composed = A.Compose(train_transform_list) if train_transform_list else None

        # Validation transforms (usually none needed unless resize wasn't done in Dataset)
        val_transforms_composed = None

    except ImportError:
        if is_main_process: _logger.warning("Albumentations not found. Skipping augmentations.")
        train_transforms_composed = None
        val_transforms_composed = None


    # 4. 创建最终的数据集实例 (using the synced slice IDs)
    train_dataset = S8DFinetune2D(
        slice_dir=data_dir,
        slice_ids=train_slice_ids, # Use broadcasted list
        transform=train_transforms_composed, # Apply augmentations
        mean=mean,
        std=std,
        img_size=img_size # Pass img_size for potential internal resize
    )
    val_dataset = S8DFinetune2D(
        slice_dir=data_dir,
        slice_ids=val_slice_ids,   # Use broadcasted list
        transform=val_transforms_composed, # Apply validation transforms (usually none)
        mean=mean,
        std=std,
        img_size=img_size # Pass img_size for potential internal resize
    )

    # 5. 创建 Samplers 和 DataLoaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if use_ddp else None


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Only shuffle if not using DDP sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Ensure drop_last=True for training with DDP
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Validation batch size can often be larger
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False # Do not drop last batch for validation
    )

    # 将 mean/std 附加到 loader 以便训练脚本访问
    train_loader.mean = mean
    train_loader.std = std
    val_loader.mean = mean
    val_loader.std = std


    return {'train': train_loader, 'val': val_loader}


# --- 本地测试 (仅在直接运行此脚本时执行) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="S8D Dataloader Test")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the s8d slice directory containing slice_manifest.csv')
    parser.add_argument('--img_size', type=int, default=8192, help='Target image size for resizing during test') # Default to 8k
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (keep low for 8k)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for testing')
    parser.add_argument('--use_ddp_test', action='store_true', help='Simulate DDP environment (single process)')
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    IMG_SIZE_TEST = args.img_size
    use_ddp_test = args.use_ddp_test

    # 模拟 DDP 环境 (单进程)
    if use_ddp_test:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501' # Use different port
        try:
             # Check if already initialized
             if not dist.is_initialized():
                 dist.init_process_group(backend='nccl')
        except Exception as e:
             print(f"Failed to initialize DDP for testing: {e}")
             use_ddp_test = False # Fallback to non-DDP test


    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, 'slice_manifest.csv')):
         print(f"\n错误: 测试数据目录 '{DATA_DIR}' 或 'slice_manifest.csv' 未找到。")
         print("请使用 --data_dir 提供您的 s8d 数据集切片目录的实际路径。")
         print("跳过本地测试。")
    else:
        try:
            print(f"\n--- 测试 build_s8d_segmentation_dataloaders (DDP={use_ddp_test}, img_size={IMG_SIZE_TEST}) ---")
            dataloaders = build_s8d_segmentation_dataloaders(
                data_dir=DATA_DIR,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                img_size=IMG_SIZE_TEST, # 测试调整大小
                use_ddp=use_ddp_test
            )
            train_loader = dataloaders['train']
            val_loader = dataloaders['val']

            print("\n--- 检查 Dataloader 输出 ---")
            if len(train_loader) == 0:
                 print("训练 Dataloader 为空，无法检查输出。请检查数据集划分或路径。")
            else:
                images, labels, slice_ids = next(iter(train_loader))
                print(f"训练批次 - Images shape: {images.shape}, type: {images.dtype}") # 应为 [B, 1, IMG_SIZE, IMG_SIZE], float32
                print(f"训练批次 - Labels shape: {labels.shape}, type: {labels.dtype}") # 应为 [B, IMG_SIZE, IMG_SIZE], int64
                print(f"训练批次 - Slice IDs: {slice_ids}")
                print(f"图像像素值范围: [{images.min():.2f}, {images.max():.2f}] (应已标准化)")
                print(f"标签值范围: [{labels.min()}, {labels.max()}] (应为类别索引)")
                print(f"Loader Mean: {train_loader.mean:.4f}, Std: {train_loader.std:.4f}")

            if len(val_loader) == 0:
                 print("\n验证 Dataloader 为空。")
            else:
                val_images, val_labels, _ = next(iter(val_loader))
                print(f"\n验证批次 - Images shape: {val_images.shape}")
                print(f"验证批次 - Labels shape: {val_labels.shape}")

            print("\nDataloader 本地测试完成!")

            # 可选：可视化第一个样本 (如果dataloader不为空)
            if len(train_loader) > 0 and len(images)>0:
                # Ensure matplotlib is imported
                try:
                    import matplotlib.pyplot as plt
                    img_vis = images[0].squeeze().numpy() # 移除批次和通道
                    lbl_vis = labels[0].numpy()
                    mean_vis, std_vis = train_loader.mean, train_loader.std
                    img_vis_denorm = (img_vis * std_vis) + mean_vis # 反归一化
                    img_vis_denorm = np.clip(img_vis_denorm, 0, 1)

                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(img_vis_denorm, cmap='gray')
                    axes[0].set_title(f"Image (Slice ID: {slice_ids[0]})")
                    axes[0].axis('off')
                    axes[1].imshow(lbl_vis, cmap='viridis', vmin=0, vmax=max(4, lbl_vis.max())) # 调整 vmax 以适应实际标签范围
                    axes[1].set_title(f"Label (Classes: {np.unique(lbl_vis)})")
                    axes[1].axis('off')
                    plt.tight_layout()
                    plt.savefig("s8d_dataloader_test_sample.png")
                    print("\n已保存第一个样本的可视化结果到 s8d_dataloader_test_sample.png")
                    plt.close()
                except ImportError:
                    print("\nMatplotlib 未安装，跳过可视化。")
                except Exception as viz_e:
                    print(f"\n可视化失败: {viz_e}")


        except FileNotFoundError as e:
            print(f"\n测试失败: {e}")
            print("请确保 --data_dir 指向正确的 s8d 切片目录，且包含 slice_manifest.csv。")
        except Exception as e:
            print(f"\n测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


    if use_ddp_test and dist.is_initialized():
        dist.destroy_process_group()

