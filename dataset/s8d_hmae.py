import os
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class S8DPretrainDataset(Dataset):
    """
    轻量化数据集：仅负责读取图像和预处理边缘图。
    复杂的四叉树分解和噪声生成将移至模型内部或训练循环。
    """
    def __init__(self, root_dir, img_size=1024):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png')) + list(self.root_dir.rglob('*.jpg'))
        self.img_size = img_size
        print(f"数据集初始化：找到 {len(self.image_files)} 张图像。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 禁用多线程 OpenCV 以避免 DataLoader 锁死
        cv2.setNumThreads(0)
        img_path = str(self.image_files[idx])

        # 1. 读取图像 (单通道)
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if full_image is None:
            full_image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 2. 统一缩放到固定尺寸 (如 1024x1024)
        if full_image.shape[0] != self.img_size or full_image.shape[1] != self.img_size:
            full_image = cv2.resize(full_image, (self.img_size, self.img_size))

        # 3. 预处理生成边缘图 (用于模型内部的四叉树分解)
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)

        # 转换为 Tensor: [1, H, W]
        img_tensor = torch.from_numpy(full_image).float().unsqueeze(0) / 255.0
        edge_tensor = torch.from_numpy(edges).float().unsqueeze(0) / 255.0

        return {
            "image": img_tensor,
            "edges": edge_tensor,
            "file_path": img_path
        }

def get_s8d_dataloader(data_root, batch_size=4, num_workers=4):
    dataset = S8DPretrainDataset(
        root_dir=data_root,
        img_size=8192
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader

if __name__ == "__main__":
    ROOT = "/work/c30636/dataset/spring8concrete/Resize_8192_output_1" 
    
    print("=== Visualizing One Batch & Statistics ===")
    BATCH_SIZE = 4
    
    loader = get_s8d_dataloader(ROOT, batch_size=BATCH_SIZE, num_workers=4)
    
    # Get one batch
    batch = next(iter(loader))
    
    images = batch['image']          # [B, 1, H, W]
    edges = batch['edges']           # [B, 1, H, W]
    file_paths = batch['file_path']  # List of file paths
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch edge tensor shape: {edges.shape}")
    print("File paths in this batch:")
    for path in file_paths:
        print(f" - {path}")
    # 计算并打印图像和边缘的统计信息
    print(f"Image stats - min: {images.min().item():.4f}, max: {images.max().item():.4f}, mean: {images.mean().item():.4f}, std: {images.std().item():.4f}")
    print(f"Edge stats  - min: {edges.min().item():.4f}, max: {edges.max().item():.4f}, mean: {edges.mean().item():.4f}, std: {edges.std().item():.4f}")          