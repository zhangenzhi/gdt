import os
import sys
import time  # 引入 time 模块
import random
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 确保 gdt.hde 路径正确
sys.path.append("./")
try:
    from gdt.hde import HierarchicalHDEProcessor
except ImportError:
    # 为了演示，如果没找到库，定义一个伪类，实际运行请忽略此块
    print("Warning: gdt.hde not found, using dummy processor for benchmark.")
    class HierarchicalHDEProcessor:
        def __init__(self, visible_fraction=0.25): pass
        def create_training_sequence(self, img, edge, fixed_length):
            # 模拟计算延迟
            time.sleep(0.5) 
            return ([np.zeros((32,32,1), dtype=np.uint8)]*fixed_length, [type('obj', (object,), {'get_coord': lambda: (0,0,10,10)})()]*fixed_length, np.zeros(fixed_length), None, None)

class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, fixed_length=1024, common_patch_size=8, verbose=False):
        self.root_dir = Path(root_dir)
        
        t_start = time.time()
        self.image_files = list(self.root_dir.rglob('*.png'))
        print(f"[Main Process] File scanning took: {time.time() - t_start:.4f}s")
        
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        self.verbose = verbose # 控制是否打印详细日志
        
        self.processor = None 
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        print(f"Dataset initialized. Found {len(self.image_files)} images.")

    def _init_processor(self):
        if self.processor is None:
            self.processor = HierarchicalHDEProcessor(visible_fraction=0.25)

    def __getitem__(self, idx):
        # 记录每一步的时间
        t0 = time.time()
        
        self._init_processor()
        img_path = str(self.image_files[idx])
        pid = os.getpid() # 获取当前进程ID，用于区分 Worker

        # 1. IO 读取
        t1 = time.time()
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        t_io = time.time() - t1

        # 2. Canny 计算
        t2 = time.time()
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        t_canny = time.time() - t2

        # 3. 维度调整
        image_input_3d = full_image[..., np.newaxis]

        # 4. HDE 算法
        t3 = time.time()
        (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
            self.processor.create_training_sequence(image_input_3d, edges, fixed_length=self.fixed_length)
        t_hde = time.time() - t3

        # 5. Patch 处理循环
        t4 = time.time()
        processed_patches = []
        coords = []
        valid_len = len(patches_seq)
        
        # 获取图像尺寸用于坐标归一化
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
            
        t_loop = time.time() - t4
        total_time = time.time() - t0

        # 只打印前几个处理或者特定的 Worker，防止刷屏
        # 在这里我们打印所有，但你可以根据 PID 过滤
        if self.verbose: 
            print(f"[Worker {pid}] Total: {total_time:.3f}s | IO: {t_io:.3f}s | Canny: {t_canny:.3f}s | HDE: {t_hde:.3f}s | PatchLoop: {t_loop:.3f}s | {img_path}")

        return {
            "pixel_values": torch.stack(processed_patches),
            "coordinates": torch.stack(coords),
            "mask": torch.tensor(is_noised_mask, dtype=torch.long)
        }

def get_hde_dataloader(data_root, batch_size=32, num_workers=32):
    # 开启 verbose 用于调试
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        fixed_length=1024,
        common_patch_size=8,
        verbose=True 
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Benchmark 时可以设为 False 保证顺序一致，但 True 模拟真实情况
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader

if __name__ == "__main__":
    ROOT = "/work/c30636/dataset/spring8concrete/Resize_8192_output_1" 
    
    # 建议先用单进程测试纯算法耗时
    # print("=== Test 1: Single Worker (Baseline Latency) ===")
    # loader_single = get_hde_dataloader(ROOT, batch_size=1, num_workers=0)
    # t_start = time.time()
    # next(iter(loader_single))
    # print(f"Single image processing time: {time.time() - t_start:.4f}s")
    # print("\n")

    print("=== Test 2: Multi Worker (Throughput) ===")
    # 注意：为了快速看到结果，可以把 batch_size 调小一点，或者 num_workers 调小
    BATCH_SIZE = 32
    NUM_WORKERS = 32
    
    print(f"Config: Batch={BATCH_SIZE}, Workers={NUM_WORKERS}")
    
    t_init_start = time.time()
    loader = get_hde_dataloader(ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    t_init_end = time.time()
    print(f"DataLoader Init time (Scanning + Setup): {t_init_end - t_init_start:.4f}s")
    
    print("Waiting for first batch...")
    t_batch_start = time.time()
    
    # 获取第一个 Batch
    batch = next(iter(loader))
    
    t_batch_end = time.time()
    print(f"First Batch Load Time: {t_batch_end - t_batch_start:.4f}s")
    print(f"Effective Throughput: {BATCH_SIZE / (t_batch_end - t_batch_start):.2f} img/s")