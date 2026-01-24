import os
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 1. 定义 Dataset (逻辑保持不变) ---
class RawImageDataset(Dataset):
    def __init__(self, root_dir, target_root, original_res, target_res):
        self.root_dir = Path(root_dir)
        self.target_root = Path(target_root)
        self.original_res = original_res
        self.target_res = target_res
        
        # 递归寻找所有 .raw 文件
        self.image_filenames = list(self.root_dir.rglob('*.raw'))
        
        if len(self.image_filenames) == 0:
            print(f"Warning: No .raw files found in {self.root_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        raw_path = self.image_filenames[idx]
        
        # 计算相对路径，保持目录结构
        relative_path = raw_path.relative_to(self.root_dir)
        save_path = self.target_root / relative_path.with_suffix('.png')
        
        try:
            # 读取 Raw 数据
            image = np.fromfile(str(raw_path), dtype=np.uint16).reshape(self.original_res, self.original_res)

            # 类型转换 / 255 -> uint8
            image = (image / 255).astype(np.uint8)

            # 如果需要缩放
            if self.original_res != self.target_res:
                image = cv2.resize(image, (self.target_res, self.target_res), interpolation=cv2.INTER_AREA)
            
            return image, str(save_path)
            
        except Exception as e:
            print(f"Error processing {raw_path}: {e}")
            # 返回 None 表示出错，需要在 collate_fn 或循环中处理，
            # 这里简单起见返回一个空标记，主循环里要做判断
            return np.array([]), ""

# --- 2. 参数解析设置 ---
def get_args():
    parser = argparse.ArgumentParser(description="Convert high-res RAW images to PNG with PyTorch Dataset.")
    
    parser.add_argument('--src_root', type=str, default="/work/c30636/dataset/spring8concrete/GC_3072_v2_16384_output_1_48_64", 
                        help='源数据根目录路径')
    
    parser.add_argument('--dst_root', type=str, default="/work/c30636/dataset/spring8concrete/Resize_8192_output_1", 
                        help='输出数据根目录路径')
    
    parser.add_argument('--original_res', type=int, default=16384, 
                        help='原始 RAW 图像的分辨率 (H=W)')
    
    parser.add_argument('--target_res', type=int, default=8192, 
                        help='目标 PNG 图像的分辨率 (H=W)')
    
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size (建议保持为 1 以便于保存文件)')
    
    parser.add_argument('--num_workers', type=int, default=32, 
                        help='并行处理的工作进程数 (建议设为 CPU 核心数)')

    return parser.parse_args()

# --- 3. 主函数 ---
def main():
    args = get_args()

    # 打印当前配置
    print("-" * 30)
    print(f"Configuration:")
    print(f"  Source:      {args.src_root}")
    print(f"  Destination: {args.dst_root}")
    print(f"  Resize:      {args.original_res} -> {args.target_res}")
    print(f"  Workers:     {args.num_workers}")
    print("-" * 30)

    # 实例化 Dataset
    dataset = RawImageDataset(
        root_dir=args.src_root, 
        target_root=args.dst_root,
        original_res=args.original_res,
        target_res=args.target_res
    )

    if len(dataset) == 0:
        return

    # 实例化 DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False
    )

    print(f"Starting conversion of {len(dataset)} files...")

    # 迭代处理
    for images, save_paths in tqdm(dataloader):
        # 此时 images 是 Tensor 或 Numpy (取决于 DataLoader 默认行为)，
        # 这里因为 transforms 没有转 tensor，所以出来是 Tensor(Byte) 或者 Double
        # 我们直接取 numpy
        
        # 遍历 batch (虽然通常 batch_size=1)
        for i in range(len(save_paths)):
            path = save_paths[i]
            img = images[i].numpy() # 转回 numpy

            # 简单的错误检查 (对应 __getitem__ 中的异常处理)
            if path == "" or img.size == 0:
                continue

            # 创建目录
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # 保存
            cv2.imwrite(path, img)

if __name__ == '__main__':
    main()