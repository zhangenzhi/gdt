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

# 假设你的库结构如下
from gdt.hde import HierarchicalHDEProcessor 

class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, image_size=8192, fixed_length=1024, common_patch_size=8):
        """
        Args:
            root_dir (str): 数据目录
            image_size (int): 必须严格匹配的图像分辨率 (8192)
            fixed_length (int): 序列最大长度
            common_patch_size (int): Patch Resize 的目标大小 (模型输入需要统一维度)
        """
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png'))
        
        self.image_size = image_size
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        
        self.processor = None 
        
        # 针对单通道灰度图的 Transform: [0, 255] -> [-1.0, 1.0]
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        print(f"HDE Dataset (Grayscale): Found {len(self.image_files)} images.")
        print(f"Strict Mode: Input MUST be {image_size}x{image_size}. No global resizing.")

    def _init_processor(self):
        if self.processor is None:
            self.processor = HierarchicalHDEProcessor(visible_fraction=0.25)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        self._init_processor()
        img_path = str(self.image_files[idx])

        # 1. 强制以灰度模式读取
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 文件损坏检查
        if full_image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        h, w = full_image.shape

        # 2. 严格的尺寸检查 (不符合直接报错)
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Invalid image resolution! Expected {self.image_size}x{self.image_size}, "
                f"but got {w}x{h} for file: {img_path}"
            )

        # 3. 边缘检测
        # 8K 图保持原分辨率处理
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # 4. 运行 HDE
        try:
            (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
                self.processor.create_training_sequence(full_image, edges, fixed_length=self.fixed_length)
        except Exception as e:
            print(f"HDE Algorithm Error on {img_path}: {e}")
            # 如果算法本身出错（比如无法生成足够的节点），可以选择跳过或者也报错
            # 这里为了训练稳健性，选择尝试读取下一张（或者你可以改成 raise e）
            return self.__getitem__((idx + 1) % len(self))

        processed_patches = []
        coords = []
        
        valid_len = len(patches_seq)

        for i in range(self.fixed_length):
            if i < valid_len:
                patch = patches_seq[i] 
                bbox = leaf_nodes[i]
                
                # 注意：虽然全图不 resize，但切出来的 patch 大小不一 (8192->4096->...->8)
                # 为了能 stack 进同一个 tensor，Patch 依然需要 resize 到 common_patch_size (8x8)
                if patch.size == 0:
                     patch_resized = np.zeros((self.common_patch_size, self.common_patch_size), dtype=np.uint8)
                else:
                    patch_resized = cv2.resize(patch, (self.common_patch_size, self.common_patch_size))
                
                # [1, 8, 8]
                patch_tensor = self.normalize(patch_resized)
                
                # 坐标归一化
                x1, x2, y1, y2 = bbox.get_coord()
                coord = torch.tensor([
                    x1 / self.image_size, 
                    y1 / self.image_size, 
                    (x2-x1) / self.image_size, 
                    (y2-y1) / self.image_size
                ], dtype=torch.float32)
                
            else:
                # Padding
                patch_tensor = torch.zeros((1, self.common_patch_size, self.common_patch_size))
                coord = torch.zeros((4,), dtype=torch.float32)
            
            processed_patches.append(patch_tensor)
            coords.append(coord)

        patches_tensor = torch.stack(processed_patches) # [Seq_Len, 1, 8, 8]
        coords_tensor = torch.stack(coords)             # [Seq_Len, 4]
        mask_tensor = torch.tensor(is_noised_mask, dtype=torch.long) # [Seq_Len]

        return {
            "pixel_values": patches_tensor,
            "coordinates": coords_tensor,
            "mask": mask_tensor
        }

def get_hde_dataloader(data_root, batch_size=32, num_workers=32):
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        image_size=8192,       # 设定预期分辨率
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
    # 配置
    ROOT = "/work/c30636/dataset/spring8concrete/Resize_8192_output_1" 
    
    # 减少 num_workers 以避免调试时的内存溢出，实际训练可调大
    loader = get_hde_dataloader(ROOT, batch_size=4, num_workers=4)
    
    print("Checking DataLoader strict mode...")
    try:
        for batch in loader:
            pixels = batch['pixel_values']
            print(f"Success! Batch Shape: {pixels.shape}")
            break
    except ValueError as e:
        print(f"Caught expected error: {e}")