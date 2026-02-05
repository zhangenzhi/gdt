import os
import sys
from torch.utils.data import DataLoader
sys.path.append("./")

import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from hmae_processor import HierarchicalMaskedAutoEncoder

class S8DPretrainDataset(Dataset):
    """
    更新后的数据集类：在 DataLoader 的 worker 进程中直接调用 HMAE 引擎进行预处理。
    该版本适配 MAE 风格架构，仅输出原始空间序列及其几何信息，掩码逻辑由模型端处理。
    """
    def __init__(self, root_dir, img_size=1024, hmae_config=None):
        self.root_dir = Path(root_dir)
        # 支持常见图像格式
        self.image_files = list(self.root_dir.rglob('*.png')) + list(self.root_dir.rglob('*.jpg'))
        self.img_size = img_size
        
        # MAE 风格下，处理器仅定义序列长度和补丁大小
        self.hmae_config = hmae_config or {
            'fixed_length': 1024,
            'patch_size': 32
        }
        self.engine = None 
        print(f"S8D 数据集初始化完成。找到 {len(self.image_files)} 张图像。")

    def _init_engine(self):
        """
        延迟初始化引擎。确保每个多进程 worker 拥有独立的引擎实例，
        避免并发冲突并确保随机性。
        """
        if self.engine is None:
            self.engine = HierarchicalMaskedAutoEncoder(
                fixed_length=self.hmae_config['fixed_length'],
                patch_size=self.hmae_config['patch_size']
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 禁用单线程中的 OpenCV 多线程，优化 DataLoader 的 worker 性能
        cv2.setNumThreads(0)
        self._init_engine()
        
        img_path = str(self.image_files[idx])
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if full_image is None:
            # 如果图像读取失败，返回一个全黑画布
            full_image = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 统一缩放至配置尺寸
        if full_image.shape[0] != self.img_size or full_image.shape[1] != self.img_size:
            full_image = cv2.resize(full_image, (self.img_size, self.img_size))

        # 1. 边缘检测：为四叉树分解提供重要性评估分值
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)

        # 2. 调用 HMAE 引擎进行 CPU 端的四叉树分解和序列生成
        # 输入形状需符合引擎要求的 (H, W, 1)
        img_np = full_image[..., np.newaxis]
        patches, coords, depths = self.engine.process_single(img_np, edges)

        # 3. 返回训练所需的全部基础数据
        return {
            "patches": patches,  # 补丁序列 [L, C, P, P]
            "targets": patches,  # 目标即为原始补丁
            "coords": coords,    # 空间坐标 [L, 4] (x1, x2, y1, y2)
            "depths": depths,    # 树深度信息 [L]
            "original_image": torch.from_numpy(full_image).float().unsqueeze(0) / 255.0 # 供可视化使用
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