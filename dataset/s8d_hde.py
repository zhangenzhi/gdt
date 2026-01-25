import os
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 假设你的 HDE 代码保存在这里，或者直接把你的类贴在上面
# from hde_core import HierarchicalHDEProcessor, Rect 

# --- 这里为了完整性，粘贴必要的简化版类定义 (确保你运行无误) ---
class Rect:
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = int(x1), int(x2), int(y1), int(y2)
    def contains(self, domain): # domain 是 edge map
        if self.y1 >= self.y2 or self.x1 >= self.x2: return 0
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch) / 255)
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2]
    def get_coord(self):
        return self.x1, self.x2, self.y1, self.y2
    def get_size(self):
        return self.x2 - self.x1, self.y2 - self.y1

# --- 数据集定义 ---
class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, crop_size=1024, fixed_length=256, common_patch_size=32):
        """
        Args:
            root_dir (str): 8K 图片的目录
            crop_size (int): 从 8K 图中切出来的训练区域大小 (建议 1024 或 512)
            fixed_length (int): 序列最大长度 (即四叉树最多分多少个块)
            common_patch_size (int): 为了送入模型，所有不同大小的块都会被 resize 到这个尺寸
        """
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png'))
        
        self.crop_size = crop_size
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        
        # 初始化你的处理器
        # 注意：这里需要在 dataset 内部实例化，或者在 worker_init_fn 中处理
        self.processor = None 
        
        # 预处理：转 Tensor 和 归一化
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"HDE Dataset: Found {len(self.image_files)} images.")
        print(f"Strategy: Random Crop {crop_size} -> Canny -> QuadTree -> Resize Patches to {common_patch_size}")

    def _init_processor(self):
        # 延迟初始化，确保多进程安全
        if self.processor is None:
            # 这里的 import 需要根据你的实际文件结构调整
            # from your_file import HierarchicalHDEProcessor
            # 这里直接使用传入的类
            self.processor = HierarchicalHDEProcessor(visible_fraction=0.25)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        self._init_processor()
        img_path = str(self.image_files[idx])

        # 1. 读取大图 (使用 OpenCV 读取，因为后续 HDE 代码也是基于 numpy/cv2)
        #    注意：OpenCV 默认是 BGR，HDE 代码看起来不依赖颜色空间，但为了统一建议转 RGB
        full_image = cv2.imread(img_path)
        if full_image is None:
            # 容错处理
            return self.__getitem__((idx + 1) % len(self))
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        
        h, w, c = full_image.shape

        # 2. 随机裁剪 (Random Crop)
        #    这是处理 8K 数据的关键。不要全图跑 Canny。
        if h > self.crop_size and w > self.crop_size:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            image_crop = full_image[top:top+self.crop_size, left:left+self.crop_size]
        else:
            # 如果图比 crop_size 小，这就 resize
            image_crop = cv2.resize(full_image, (self.crop_size, self.crop_size))

        # 3. 生成 Edge Map (Canny)
        #    参数需要根据你的数据特点微调
        gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # 4. 运行 HDE 核心逻辑
        #    注意：fixed_length 控制序列长度
        try:
            (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
                self.processor.create_training_sequence(image_crop, edges, fixed_length=self.fixed_length)
        except Exception as e:
            print(f"HDE Error on {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 5. 后处理数据以适配 Batch
        #    我们需要返回：
        #    - pixel_values: [Seq_Len, 3, 32, 32] (统一尺寸后的图像块)
        #    - coordinates:  [Seq_Len, 4] (x1, y1, w, h) (用于位置编码)
        #    - mask:         [Seq_Len] (0=visible, 1=masked/noised, -1=padding)
        
        processed_patches = []
        coords = []
        
        valid_len = len(patches_seq)

        for i in range(self.fixed_length):
            if i < valid_len:
                patch = patches_seq[i] # numpy array (H, W, 3)
                bbox = leaf_nodes[i]
                
                # Resize 到统一尺寸 (例如 32x32) 以便 stack
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                     patch_resized = np.zeros((self.common_patch_size, self.common_patch_size, 3), dtype=np.uint8)
                else:
                    patch_resized = cv2.resize(patch, (self.common_patch_size, self.common_patch_size))
                
                # 转 Tensor 并归一化 -> [3, 32, 32]
                # ToTensor 会把 (H,W,C) [0-255] 转为 (C,H,W) [0.0-1.0]
                patch_tensor = self.normalize(patch_resized)
                
                # 记录归一化的坐标 (x, y, w, h) / crop_size
                x1, x2, y1, y2 = bbox.get_coord()
                coord = torch.tensor([
                    x1 / self.crop_size, 
                    y1 / self.crop_size, 
                    (x2-x1) / self.crop_size, 
                    (y2-y1) / self.crop_size
                ], dtype=torch.float32)
                
            else:
                # Padding 部分
                patch_tensor = torch.zeros((3, self.common_patch_size, self.common_patch_size))
                coord = torch.zeros((4,), dtype=torch.float32)
            
            processed_patches.append(patch_tensor)
            coords.append(coord)

        # 堆叠成 Tensor
        patches_tensor = torch.stack(processed_patches) # [Seq_Len, 3, 32, 32]
        coords_tensor = torch.stack(coords)             # [Seq_Len, 4]
        mask_tensor = torch.tensor(is_noised_mask, dtype=torch.long) # [Seq_Len]

        return {
            "pixel_values": patches_tensor,
            "coordinates": coords_tensor,
            "mask": mask_tensor
        }

# --- DataLoader 创建函数 ---
def get_hde_dataloader(data_root, batch_size=16, num_workers=8):
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        crop_size=8192,      # 在 8192x8192 的区域内构建四叉树
        fixed_length=1024,    # 序列长度限制为 1024 个 patch
        common_patch_size=8 # 即使 patch 只有 8x8，也 resize 到 8x8 喂给模型
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
    ROOT = "/work/c30636/dataset/spring8concrete/Resize_8192_output_1" # 你的数据路径
    
    # 获取 Loader
    loader = get_hde_dataloader(ROOT, batch_size=4, num_workers=4)
    
    print("开始测试 DataLoader...")
    for batch in loader:
        pixels = batch['pixel_values']
        coords = batch['coordinates']
        mask = batch['mask']
        
        print(f"Batch Pixel Shape: {pixels.shape}") 
        # 预期: [4, 256, 3, 32, 32] -> [Batch, Seq_Len, Channel, H, W]
        
        print(f"Batch Coords Shape: {coords.shape}") 
        # 预期: [4, 256, 4]
        
        print(f"Batch Mask Shape:  {mask.shape}")  
        # 预期: [4, 256]
        
        print(f"Example Mask Values: {mask[0][:10]}")
        break