

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Mean and Std for S8D Dataset")
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/s8d/pretrain", help='Path to the S8D dataset directory')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for DataLoader')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for calculation')
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

