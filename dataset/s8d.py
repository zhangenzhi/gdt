

import os
import sys
sys.path.append("./")
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from PIL import Image, ImageFile
import tifffile

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Spring8Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading Spring8 .raw image files.
    Outputs single-channel, normalized tensors.
    """
    def __init__(self, data_path, resolution, file_list):
        self.data_path = data_path
        self.resolution = resolution
        self.image_filenames = file_list
        
        # Define transformations for single-channel images
        # NOTE: For best results, calculate the true mean and std of your dataset
        # and replace the [0.5] values below with the calculated ones.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # 例如: transforms.Normalize(mean=[0.123], std=[0.456])
            transforms.Normalize(mean=[0.390480], std=[0.29985])
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Construct the full path to the image file
        img_path = self.image_filenames[idx]
        
        # Load the raw uint16 data and reshape
        image = np.fromfile(img_path, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        
        # Convert from uint16 (0-65535) to uint8 (0-255) range while preserving dynamic range
        image = (image / 65535.0 * 255.0).astype(np.uint8)
        
        # Apply transforms (ToTensor, Normalize)
        # The output will be a tensor of shape [1, resolution, resolution]
        image_tensor = self.transform(image)
        
        return image_tensor

def build_s8d_dataloaders(data_dir, batch_size, num_workers, resolution=8192, val_split=0.1):
    """
    Builds and returns train and validation DataLoaders for the Spring8Dataset.
    Handles DDP (Distributed Data Parallel) samplers.
    """
    print(f"Building Spring8 dataloaders from directory: {data_dir}")
    
    # Discover all .raw files recursively
    all_files = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".raw"):
                all_files.append(os.path.join(root, name))
                
    if not all_files:
        raise FileNotFoundError(f"No .raw files found in {data_dir}. Please check the path.")

    print(f"Found {len(all_files)} total .raw image files.")

    # Split the dataset into training and validation sets
    num_files = len(all_files)
    num_val = int(num_files * val_split)
    num_train = num_files - num_val

    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_files, val_files = random_split(all_files, [num_train, num_val], generator=generator)

    # Create dataset instances
    train_dataset = Spring8Dataset(data_dir, resolution, list(train_files))
    val_dataset = Spring8Dataset(data_dir, resolution, list(val_files))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DDP samplers if in a distributed environment
    is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Shuffle only if not using DDP sampler
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return {'train': train_loader, 'val': val_loader}


# --- Script to calculate dataset statistics ---

class Spring8DatasetForStats(Dataset):
    """A temporary dataset that only converts images to tensors between [0, 1]."""
    def __init__(self, all_files, resolution):
        self.resolution = resolution
        self.image_filenames = all_files
        self.transform = transforms.ToTensor() # Only convert to tensor, NO normalization

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        image = np.fromfile(img_path, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        image = (image / 65535.0 * 255.0).astype(np.uint8)
        # ToTensor scales uint8 (0-255) to float (0.0-1.0), which is what we need
        return self.transform(image)

class S8DFinetune2D(Dataset):
    """PyTorch Dataset for loading 2D slices from .tiff files."""
    
    def __init__(self, slice_dir, num_classes=5, transform=None, target_transform=None, subset=None):
        self.slice_dir = slice_dir
        self.transform = transform
        self.num_classes = num_classes
        self.target_transform = target_transform
        self.manifest = self._load_manifest()

        if subset is not None:
            self.manifest = self.manifest[self.manifest['slice_id'].isin(subset)].reset_index(drop=True)
        
    def _load_manifest(self):
        manifest_path = os.path.join(self.slice_dir, 'slice_manifest.csv')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到清单文件: {manifest_path}。请确保该文件存在于您的数据目录中。")
        return pd.read_csv(manifest_path)
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        record = self.manifest.iloc[idx]
        
        # 加载图像和标签
        img_path = os.path.join(self.slice_dir, record['image_path'])
        label_path = os.path.join(self.slice_dir, record['label_path'])
        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)
        
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        
        # 转换为张量
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        label_tensor = torch.from_numpy(label.astype(np.int64)) #<-- 关键修改: 直接返回 LongTensor
        
        # 归一化图像
        min_val, max_val = torch.min(img_tensor), torch.max(img_tensor)
        img_tensor = (img_tensor - min_val) / (max_val - min_val + 1e-6)
        
        return img_tensor, label_tensor, record['slice_id']
    
    def get_volume_ids(self):
        return sorted(self.manifest['volume_id'].unique())
    
    def get_slices_for_volume(self, volume_id):
        return self.manifest[self.manifest['volume_id'] == volume_id]['slice_id'].tolist()

def build_s8d_segmentation_dataloaders(data_dir, batch_size, num_workers):
    """
    创建数据加载器，并自动进行 80/20 训练/验证集划分。
    """
    # 实例化一次以读取 manifest 和 volume_ids
    full_dataset = S8DFinetune2D(slice_dir=data_dir)
    volume_ids = full_dataset.get_volume_ids()
    
    # 基于 volume_id 划分训练和验证集
    np.random.shuffle(volume_ids)
    split_idx = int(len(volume_ids) * 0.8)
    train_volume_ids = volume_ids[:split_idx]
    val_volume_ids = volume_ids[split_idx:]
    
    train_slice_ids = [s for vol_id in train_volume_ids for s in full_dataset.get_slices_for_volume(vol_id)]
    val_slice_ids = [s for vol_id in val_volume_ids for s in full_dataset.get_slices_for_volume(vol_id)]

    print(f"数据集划分: {len(train_slice_ids)} 个训练切片, {len(val_slice_ids)} 个验证切片。")

    # 使用划分好的子集创建数据集实例
    train_dataset = S8DFinetune2D(slice_dir=data_dir, subset=train_slice_ids)
    val_dataset = S8DFinetune2D(slice_dir=data_dir, subset=val_slice_ids)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    
    return {'train': train_loader, 'val': val_loader}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Mean and Std for S8D Dataset")
    # parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/s8d/pretrain", help='Path to the S8D dataset directory')
    parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/spring8data/8192_output_1/No_001", help='Path to the S8D dataset directory')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for DataLoader')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for calculation')
    parser.add_argument('--resolution', type=int, default=8192, help='Image resolution')
    args = parser.parse_args()
    
    print("Discovering all .raw files for statistics calculation...")
    all_files = []
    for root, _, files in os.walk(args.data_dir):
        for name in files:
            if name.endswith(".raw"):
                all_files.append(os.path.join(root, name))

    if not all_files:
        raise FileNotFoundError(f"No .raw files found in {args.data_dir}. Please check the path.")

    print(f"Found {len(all_files)} files. Creating DataLoader...")
    
    # Use the special dataset for calculation
    stats_dataset = Spring8DatasetForStats(all_files, args.resolution)
    # No need for a sampler, we want to iterate through all files once
    stats_loader = DataLoader(
        stats_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Calculate mean and std
    channels_sum = 0.
    channels_squared_sum = 0.
    num_pixels = 0

    for images in tqdm(stats_loader, desc="Calculating Mean and Std"):
        # images shape: [B, C, H, W]
        # Important: Move to float64 for precision during summation
        images = images.to(torch.float64)
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        channels_sum += torch.sum(images, dim=[0, 2, 3])
        channels_squared_sum += torch.sum(images**2, dim=[0, 2, 3])

    mean = channels_sum / num_pixels
    std = (channels_squared_sum / num_pixels - mean ** 2) ** 0.5

    print("\n" + "="*40)
    print("Calculation Complete!")
    print(f"  - Total Pixels: {num_pixels}")
    print(f"  - Mean: {mean.tolist()}")
    print(f"  - Std:  {std.tolist()}")
    print("="*40)
    print("\nPlease copy these Mean and Std values into the transforms.Normalize")
    print("function in the Spring8Dataset class in dataset/s8d.py")

