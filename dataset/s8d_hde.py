import os
import sys
sys.path.append("./")
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 确保 gdt.hde 路径正确
from gdt.hde import HierarchicalHDEProcessor 

class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, fixed_length=1024, common_patch_size=8):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png'))
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        
        # 懒加载 Processor
        self.processor = None 
        
        # 归一化: [0, 255] -> [-1.0, 1.0]
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        print(f"Dataset initialized. Found {len(self.image_files)} images.")

    def _init_processor(self):
        if self.processor is None:
            self.processor = HierarchicalHDEProcessor(visible_fraction=0.25)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        self._init_processor()
        img_path = str(self.image_files[idx])

        # 1. 读取图像 (灰度模式) -> (H, W)
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 生成 Edge Map (Canny 需要 2D 输入)
        #    如果 full_image 是 None 这里会直接报 AttributeError，符合你的要求
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # 3. 【关键修复】: 升维 (H, W) -> (H, W, 1)
        #    HDE 算法里写了 h,w,c = img.shape，所以必须给它 3 个维度
        image_input_3d = full_image[..., np.newaxis]

        # 4. 运行 HDE (不加 try-except，出错直接崩)
        (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
            self.processor.create_training_sequence(image_input_3d, edges, fixed_length=self.fixed_length)

        processed_patches = []
        coords = []
        
        # 这里的 patches_seq 里的 patch 也是 (H, W, 1) 形状
        valid_len = len(patches_seq)

        for i in range(self.fixed_length):
            if i < valid_len:
                patch = patches_seq[i] 
                bbox = leaf_nodes[i]
                
                # Resize: 输入 (H, W, 1) -> 输出 (8, 8) (OpenCV 会自动降维)
                # 如果 patch 为空，这里 cv2.resize 会报错，符合要求
                patch_resized = cv2.resize(patch, (self.common_patch_size, self.common_patch_size))
                
                # ToTensor 会自动加一个通道维度: (8, 8) -> (1, 8, 8)
                patch_tensor = self.normalize(patch_resized)
                
                # 坐标归一化 (注意：这里假设原图是 8192，如果不一致坐标会偏，但不检查)
                # 使用 shape 获取真实宽高
                h_img, w_img = full_image.shape
                x1, x2, y1, y2 = bbox.get_coord()
                
                coord = torch.tensor([
                    x1 / w_img, 
                    y1 / h_img, 
                    (x2-x1) / w_img, 
                    (y2-y1) / h_img
                ], dtype=torch.float32)
                
            else:
                # Padding
                patch_tensor = torch.zeros((1, self.common_patch_size, self.common_patch_size))
                coord = torch.zeros((4,), dtype=torch.float32)
            
            processed_patches.append(patch_tensor)
            coords.append(coord)

        return {
            "pixel_values": torch.stack(processed_patches),
            "coordinates": torch.stack(coords),
            "mask": torch.tensor(is_noised_mask, dtype=torch.long)
        }

def get_hde_dataloader(data_root, batch_size=32, num_workers=32):
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        fixed_length=1024,
        common_patch_size=8
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
    
    # 测试运行
    loader = get_hde_dataloader(ROOT, batch_size=32, num_workers=32)
    
    print("Starting DataLoader (No Safety Checks)...")
    for batch in loader:
            pixels = batch['pixel_values']
            coords = batch['coordinates']
            mask = batch['mask']
            
            # 预期输出形状
            print(f"Batch Pixel Shape: {pixels.shape}") 
            # 预期: [32, 1024, 1, 8, 8]  (Batch, Seq, Channel, H, W)
            
            print(f"Batch Coords Shape: {coords.shape}") 
            # 预期: [32, 1024, 4]
            
            print(f"Batch Mask Shape:  {mask.shape}")  
            # 预期: [32, 1024]
            
            # 检查数据范围是否归一化到了 [-1, 1] 附近
            print(f"Pixel Range: min={pixels.min():.2f}, max={pixels.max():.2f}")
            break