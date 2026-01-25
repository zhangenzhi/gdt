import os
import sys
import time
import random
import cv2
import numpy as np
import torch
import multiprocessing
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 确保 gdt.hde 路径正确
sys.path.append("./")
try:
    from gdt.hde import HierarchicalHDEProcessor
except ImportError:
    print("Warning: gdt.hde not found, using dummy processor for benchmark.")
    class HierarchicalHDEProcessor:
        def __init__(self, visible_fraction=0.25): pass
        def create_training_sequence(self, img, edge, fixed_length):
            time.sleep(0.1) 
            return ([np.zeros((32,32,1), dtype=np.uint8)]*fixed_length, [type('obj', (object,), {'get_coord': lambda: (0,0,10,10)})()]*fixed_length, np.zeros(fixed_length), None, None)

class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, fixed_length=1024, common_patch_size=8, counter=None):
        self.root_dir = Path(root_dir)
        
        t_start = time.time()
        self.image_files = list(self.root_dir.rglob('*.png'))
        print(f"[Main Process] File scanning took: {time.time() - t_start:.4f}s")
        
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        self.counter = counter 
        
        self.processor = None 
        
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
        # 禁止 OpenCV 在子进程中多线程，防止与 DataLoader 的多进程冲突导致死锁或性能下降
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # 计数器累加
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
        
        self._init_processor()
        img_path = str(self.image_files[idx])

        # 1. IO 读取
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 2. Canny 计算
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)

        # 3. 维度调整
        image_input_3d = full_image[..., np.newaxis]

        # 4. HDE 算法
        (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
            self.processor.create_training_sequence(image_input_3d, edges, fixed_length=self.fixed_length)

        # 5. Patch 处理循环
        processed_patches = []
        coords = []
        valid_len = len(patches_seq)
        
        h_img, w_img = full_image.shape

        for i in range(self.fixed_length):
            if i < valid_len:
                patch = patches_seq[i] 
                bbox = leaf_nodes[i]
                
                # Resize
                patch_resized = cv2.resize(patch, (self.common_patch_size, self.common_patch_size))
                patch_tensor = self.normalize(patch_resized)
                
                x1, x2, y1, y2 = bbox.get_coord()
                coord = torch.tensor([
                    x1 / w_img, y1 / h_img, (x2-x1) / w_img, (y2-y1) / h_img
                ], dtype=torch.float32)
            else:
                patch_tensor = torch.zeros((1, self.common_patch_size, self.common_patch_size))
                coord = torch.zeros((4,), dtype=torch.float32)
            
            processed_patches.append(patch_tensor)
            coords.append(coord)

        return {
            "pixel_values": torch.stack(processed_patches),
            "coordinates": torch.stack(coords),
            "mask": torch.tensor(is_noised_mask, dtype=torch.long)
        }

def get_hde_dataloader(data_root, batch_size=32, num_workers=4, prefetch_factor=2, counter=None):
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        fixed_length=1024,
        common_patch_size=8,
        counter=counter
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        drop_last=True
    )
    return loader

if __name__ == "__main__":
    ROOT = "/work/c30636/dataset/spring8concrete/Resize_8192_output_1" 
    
    print("=== Test: Full Epoch Benchmark ===")
    BATCH_SIZE = 32
    NUM_WORKERS = 32 
    PREFETCH = 2
    
    processed_counter = multiprocessing.Value('i', 0)

    print(f"Config: Batch={BATCH_SIZE}, Workers={NUM_WORKERS}, Prefetch={PREFETCH}")
    
    # 初始化 DataLoader
    t_init_start = time.time()
    loader = get_hde_dataloader(ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, counter=processed_counter)
    t_init_end = time.time()
    print(f"DataLoader Init time: {t_init_end - t_init_start:.4f}s")
    
    print(f"Starting to iterate over {len(loader)} batches...")
    
    t_epoch_start = time.time()
    
    # 遍历整个 Epoch
    for i, batch in enumerate(loader):
        # 每 10 个 Batch 打印一次进度，避免无聊
        if i % 10 == 0:
            elapsed = time.time() - t_epoch_start
            imgs_done = (i + 1) * BATCH_SIZE
            speed = imgs_done / elapsed if elapsed > 0 else 0
            print(f"Batch [{i}/{len(loader)}] - Speed: {speed:.2f} img/s - Total processed (workers): {processed_counter.value}")
        time.sleep(4)  # 模拟一些处理时间

    t_epoch_end = time.time()
    total_time = t_epoch_end - t_epoch_start
    total_images_dataset = len(loader) * BATCH_SIZE # 近似值 (因为 drop_last=True)
    
    print("-" * 30)
    print(f"Full Epoch Finished!")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average Throughput: {total_images_dataset / total_time:.2f} img/s")
    print(f"Total images processed by workers: {processed_counter.value}")
    print("-" * 30)