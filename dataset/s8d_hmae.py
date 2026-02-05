import os
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from gdt.hmae import HierarchicalMaskedAutoEncoder

class S8DPretrainDataset(Dataset):
    """
    更新后的数据集：在 worker 进程中直接调用 HMAE 引擎进行预处理。
    """
    def __init__(self, root_dir, img_size=1024, hmae_config=None):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png')) + list(self.root_dir.rglob('*.jpg'))
        self.img_size = img_size
        
        # 在这里初始化引擎配置，但实际引擎实例建议在 getitem 或 worker_init 中创建
        self.hmae_config = hmae_config or {
            'visible_fraction': 0.25,
            'fixed_length': 1024,
            'patch_size': 32
        }
        self.engine = None 
        print(f"Dataset initialized. Found {len(self.image_files)} images.")

    def _init_engine(self):
        # 延迟初始化，确保每个 worker 进程有自己的引擎实例（避免随机种子冲突）
        if self.engine is None:
            self.engine = HierarchicalMaskedAutoEncoder(
                visible_fraction=self.hmae_config['visible_fraction'],
                fixed_length=self.hmae_config['fixed_length'],
                patch_size=self.hmae_config['patch_size']
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        self._init_engine()
        
        img_path = str(self.image_files[idx])
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if full_image is None:
            full_image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        if full_image.shape[0] != self.img_size or full_image.shape[1] != self.img_size:
            full_image = cv2.resize(full_image, (self.img_size, self.img_size))

        # 1. 边缘检测
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)

        # 2. 调用 HMAE 引擎进行复杂的 CPU 预处理
        # 注意：这里我们处理的是单张图 [H, W, 1]
        img_np = full_image[..., np.newaxis]
        p, t, n, c, d, m = self.engine.process_single(img_np, edges)

        # 返回模型所需的所有张量
        return {
            "patches": p,      # [L, C, P, P]
            "targets": t,      # [L, C, P, P]
            "noises": n,       # [L, C, P, P]
            "coords": c,       # [L, 4]
            "depths": d,       # [L]
            "mask": m,         # [L]
            "original_image": torch.from_numpy(full_image).float().unsqueeze(0) / 255.0 # 仅供可视化使用
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