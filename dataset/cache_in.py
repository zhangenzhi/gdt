import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import math
import time
import argparse
import json

# =============================================================================
# Section 1: Cache Building Logic
# =============================================================================

def build_cache_worker(args):
    """
    单个工作进程的任务：读取一部分图片，处理后写入到共享的 np.memmap 数组中。
    """
    image_paths, memmap_path, shape, dtype, indices_range = args
    
    # 在工作进程中重新打开 memmap 文件
    images_memmap = np.memmap(memmap_path, dtype=dtype, mode='r+', shape=shape)

    start, end = indices_range
    for i in range(start, end):
        img_path = image_paths[i]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224), Image.Resampling.BILINEAR)
            
            # 将数据写入 memmap 数组的指定位置
            images_memmap[i] = np.array(img)
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
            # 可选：出错时填充黑色图像
            images_memmap[i] = np.zeros((224, 224, 3), dtype=dtype)

    return True

def build_cache_in_parallel(dataset, cache_path, num_workers=8):
    """
    使用多进程并行构建数据集缓存。
    """
    image_cache_path = cache_path
    label_cache_path = cache_path.replace(".mmap", "_labels.npy")
    meta_cache_path = cache_path.replace(".mmap", "_meta.json")

    if os.path.exists(image_cache_path) and os.path.exists(label_cache_path) and os.path.exists(meta_cache_path):
        print(f"缓存文件已存在于 {cache_path}。跳过构建。")
        return

    # dataset.samples 是一个 (path, label) 的列表
    image_paths = [s[0] for s in dataset.samples]
    labels = [s[1] for s in dataset.samples]
    num_samples = len(dataset)
    shape = (num_samples, 224, 224, 3)
    dtype = 'uint8'

    # 1. 保存元数据文件，使其更健壮
    meta_data = {'shape': shape, 'dtype': str(np.dtype(dtype))}
    with open(meta_cache_path, 'w') as f:
        json.dump(meta_data, f)

    print(f"正在于 {image_cache_path} 创建缓存文件，形状为 {shape}")
    # 2. 为图像创建空的 memmap 文件
    images_memmap = np.memmap(image_cache_path, dtype=dtype, mode='w+', shape=shape)
    # 将标签保存到一个独立的 numpy 文件中
    np.save(label_cache_path, np.array(labels))
    del images_memmap # 关闭文件句柄

    # 3. 分配任务给工作进程
    chunk_size = math.ceil(num_samples / num_workers)
    tasks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_samples)
        if start >= end: continue
        
        # 每个任务包含所有必要信息
        task_args = (image_paths, image_cache_path, shape, dtype, (start, end))
        tasks.append(task_args)

    # 4. 使用进程池并行执行
    print(f"使用 {num_workers} 个工作进程构建缓存...")
    # 为跨平台兼容性，设置 start_method 为 'spawn'
    with mp.get_context("spawn").Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(build_cache_worker, tasks), total=len(tasks)))
    
    print("缓存构建完成。")

# =============================================================================
# Section 2: Cached Dataset Class
# =============================================================================

class NpMemmapDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        self.transform = transform
        
        image_cache_path = cache_path
        label_cache_path = cache_path.replace(".mmap", "_labels.npy")
        meta_cache_path = cache_path.replace(".mmap", "_meta.json")
        
        if not all(os.path.exists(p) for p in [image_cache_path, label_cache_path, meta_cache_path]):
            raise FileNotFoundError(f"缓存文件未找到。请先构建缓存。在 {os.path.dirname(cache_path)} 中搜索")

        # 从元数据文件加载形状和数据类型
        with open(meta_cache_path, 'r') as f:
            meta_data = json.load(f)
        shape = tuple(meta_data['shape'])
        dtype = np.dtype(meta_data['dtype'])

        self.labels = np.load(label_cache_path)
        self.images = np.memmap(image_cache_path, dtype=dtype, mode='r', shape=shape)

        if shape[0] != len(self.labels):
            raise ValueError("图像缓存和标签缓存中的样本数量不匹配！")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_np = self.images[index]
        label = self.labels[index]
        img_pil = Image.fromarray(img_np)
        
        if self.transform:
            img_pil = self.transform(img_pil)
        
        return img_pil, torch.tensor(label, dtype=torch.long)

# =============================================================================
# Section 3: Main Execution and Testing Block
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImageNet Memmap Cache Test Script for Train and Val')
    parser.add_argument('--data_dir', type=str, default='/work/c30636/dataset/imagenet/', help='原始ImageNet数据集目录的路径 (包含 train/ 和 val/ 子目录)')
    parser.add_argument('--cache_dir', type=str, default='/work/c30636/dataset/imagenet_cache_64/', help='用于存储生成的缓存文件 (.mmap, .npy, .json) 的目录')
    parser.add_argument('--num_workers', type=int, default=16, help='用于构建缓存和加载数据的工作进程数')
    parser.add_argument('--batch_size', type=int, default=4096, help='DataLoader 测试的批次大小')
    parser.add_argument('--force_rebuild', action='store_true', help='即使缓存已存在，也强制重建')
    parser.add_argument('--test_epochs', type=int, default=8, help='用于 DataLoader 性能测试的 epoch 数')
    args = parser.parse_args()

    # --- 设置路径和目录 ---
    os.makedirs(args.cache_dir, exist_ok=True)
    splits_to_process = ['train', 'val']

    # --- 阶段 1: 为所有数据子集构建缓存 (带计时) ---
    print("\n" + "="*50)
    print("阶段 1: 缓存构建")
    print("="*50)
    
    for split in splits_to_process:
        print(f"\n--- 正在处理子集: {split} ---")
        original_data_dir = os.path.join(args.data_dir, split)
        cache_path = os.path.join(args.cache_dir, f'{split}_cache.mmap')

        if not os.path.isdir(original_data_dir):
            print(f"警告: 未找到子集 '{split}' 的目录: {original_data_dir}。跳过。")
            continue
        
        # 仅当缓存不存在或被强制时才构建
        if args.force_rebuild or not os.path.exists(cache_path):
            print(f"缓存未找到或被强制重建。开始构建过程...")
            original_dataset = ImageFolder(original_data_dir)
            
            build_start_time = time.time()
            build_cache_in_parallel(original_dataset, cache_path, num_workers=args.num_workers)
            build_end_time = time.time()
            
            print(f"\n--- '{split}' 子集缓存构建时间: {build_end_time - build_start_time:.2f} 秒 ---")
        else:
            print(f"找到 '{split}' 子集的缓存。跳过构建过程。")

    # --- 阶段 2: 测试 DataLoader (带计时) ---
    print("\n" + "="*50)
    print("阶段 2: DATALOADER 性能测试")
    print("="*50)

    # 为训练和验证定义不同的数据增强
    transforms = {
        'train': T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    
    for split in splits_to_process:
        print(f"\n--- 正在测试子集: {split} ---")
        cache_path = os.path.join(args.cache_dir, f'{split}_cache.mmap')

        if not os.path.exists(cache_path):
            print(f"未找到 '{split}' 子集的缓存。跳过测试。")
            continue

        cached_dataset = NpMemmapDataset(cache_path, transform=transforms[split])
        data_loader = DataLoader(
            cached_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=(split == 'train'), # 仅在训练时打乱
            pin_memory=True
        )

        print(f"为 '{split}' 创建 DataLoader，批次大小={args.batch_size}，工作进程数={args.num_workers}...")
        
        num_epochs_to_test = args.test_epochs
        print(f"开始对 {num_epochs_to_test} 个 epoch 进行迭代测试...")

        total_duration = 0
        total_images_processed = 0

        # --- 迭代多个 epoch 以获得平均值 ---
        for epoch in range(num_epochs_to_test):
            print(f"  --- Epoch {epoch + 1}/{num_epochs_to_test} ---")
            epoch_start_time = time.time()
            
            # 使用 tqdm 来显示进度条，提供更好的视觉反馈
            progress_bar = tqdm(data_loader, desc=f"  Epoch {epoch+1}", leave=False, unit="batch")
            for i, (images, labels) in enumerate(progress_bar):
                # 在真实的训练循环中，这里会执行模型的前向/反向传播
                # 为了纯粹测试数据加载速度，我们只迭代
                pass

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_images = len(cached_dataset)
            epoch_throughput = epoch_images / epoch_duration if epoch_duration > 0 else 0

            total_duration += epoch_duration
            total_images_processed += epoch_images
            
            print(f"  Epoch {epoch + 1} 耗时: {epoch_duration:.2f} 秒，吞吐量: {epoch_throughput:.2f} 张/秒")

        # --- 计算并打印最终的平均结果 ---
        average_epoch_time = total_duration / num_epochs_to_test if num_epochs_to_test > 0 else 0
        average_throughput = total_images_processed / total_duration if total_duration > 0 else 0
        
        print(f"\n--- '{split}' 子集 DataLoader 测试结果 ---")
        print(f"完成 {num_epochs_to_test} 个 epoch 的总耗时: {total_duration:.2f} 秒")
        print(f"平均每个 epoch 耗时: {average_epoch_time:.2f} 秒")
        print(f"平均图像处理吞吐量: {average_throughput:.2f} 张/秒")

