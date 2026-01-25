import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure gdt.hde path is correct
sys.path.append("./")
from gdt.hde import HierarchicalHDEProcessor

class HDEPretrainDataset(Dataset):
    def __init__(self, root_dir, fixed_length=1024, common_patch_size=8):
        self.root_dir = Path(root_dir)
        self.image_files = list(self.root_dir.rglob('*.png'))
        
        self.fixed_length = fixed_length
        self.common_patch_size = common_patch_size
        
        self.processor = None 
        
        # Only ToTensor, NO Normalize (to keep values in 0-1 range for viz)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

        print(f"Dataset initialized. Found {len(self.image_files)} images.")

    def _init_processor(self):
        if self.processor is None:
            self.processor = HierarchicalHDEProcessor(visible_fraction=0.5)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Force single thread for OpenCV in workers
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        
        self._init_processor()
        img_path = str(self.image_files[idx])

        # 1. Read Image
        full_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if full_image is None:
            # Fallback for broken images if necessary
            full_image = np.zeros((8192, 8192), dtype=np.uint8)

        # 2. Pre-processing
        blurred = cv2.GaussianBlur(full_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 100)
        image_input_3d = full_image[..., np.newaxis] # (H, W, 1)

        # 3. Run HDE Algorithm
        (patches_seq, leaf_nodes, is_noised_mask, _, _) = \
            self.processor.create_training_sequence(image_input_3d, edges, fixed_length=self.fixed_length)

        # 4. Prepare Batch Data
        processed_patches = []
        coords = []
        valid_len = len(patches_seq)
        
        h_img, w_img = full_image.shape

        for i in range(self.fixed_length):
            if i < valid_len:
                patch = patches_seq[i] 
                bbox = leaf_nodes[i]
                
                # Resize Patch to common size (e.g., 8x8)
                if patch.shape[0] > 0 and patch.shape[1] > 0:
                    patch_resized = cv2.resize(patch, (self.common_patch_size, self.common_patch_size))
                else:
                    patch_resized = np.zeros((self.common_patch_size, self.common_patch_size), dtype=np.uint8)

                # Normalize (0-1 float)
                patch_tensor = self.normalize(patch_resized)
                
                # Normalize Coordinates (0-1)
                x1, x2, y1, y2 = bbox.get_coord()
                coord = torch.tensor([
                    x1 / w_img, y1 / h_img, (x2-x1) / w_img, (y2-y1) / h_img
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
            "mask": torch.tensor(is_noised_mask, dtype=torch.long),
            "file_path": img_path  # Return path to load original image in main
        }

def get_hde_dataloader(data_root, batch_size=4, num_workers=4):
    dataset = HDEPretrainDataset(
        root_dir=data_root,
        fixed_length=1024,
        common_patch_size=32
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
    
    loader = get_hde_dataloader(ROOT, batch_size=BATCH_SIZE, num_workers=4)
    
    # Get one batch
    batch = next(iter(loader))
    
    pixel_values = batch['pixel_values'] # [B, Seq, 1, 8, 8]
    coordinates = batch['coordinates']   # [B, Seq, 4]
    masks = batch['mask']                # [B, Seq]
    file_paths = batch['file_path']      # list of paths
    
    print(f"Batch Loaded. Visualizing first sample and calculating stats...")
    
    # --- Visualization Parameters ---
    VIS_SIZE = 8192 # Size of the visualization canvas
    sample_idx = 0
    
    # 1. Load Original Image & Generate Edges
    orig_path = file_paths[sample_idx]
    
    # 读取原始全分辨率图像
    full_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    
    if full_img is None:
        print("Error loading original image, using black canvas.")
        original_img = np.zeros((VIS_SIZE, VIS_SIZE), dtype=np.uint8)
        edges_vis = np.zeros((VIS_SIZE, VIS_SIZE), dtype=np.uint8)
    else:
        # 复现 Dataset 中的预处理步骤 (在全分辨率上计算 Canny)
        blurred_full = cv2.GaussianBlur(full_img, (5, 5), 0)
        edges_full = cv2.Canny(blurred_full, 50, 100)
        
        # 【新增】: 膨胀边缘，防止在 resize 时消失
        # 8K图片线条太细，resize到2K时如果不加粗，插值后会变成几乎看不见的灰色
        kernel = np.ones((3,3), np.uint8) # 5x5 核意味着线条会被加粗到 ~5px
        edges_dilated = cv2.dilate(edges_full, kernel, iterations=1)
        
        # 缩放到可视化尺寸
        original_img = cv2.resize(full_img, (VIS_SIZE, VIS_SIZE))
        # 使用最近邻插值保持线条锐利
        edges_vis = cv2.resize(edges_dilated, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # 2. Reconstruct "Model Input" from Patches
    # This shows exactly what the Transformer sees (patches resized to 8x8, some noised)
    reconstructed_input = np.zeros((VIS_SIZE, VIS_SIZE), dtype=np.uint8)
    
    # 3. Create Mask Visualization (Red overlay for noised areas)
    mask_vis = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    seq_len = pixel_values.shape[1]
    
    # 用于统计的列表
    all_patch_widths = []
    
    # ---------------------------
    # Visualization & Stats Loop
    # ---------------------------
    for b in range(BATCH_SIZE): # 遍历 Batch 中的所有图片进行统计
        for i in range(seq_len):
            # 获取 coordinates (normalized 0-1)
            x_norm, y_norm, w_norm, h_norm = coordinates[b, i].tolist()
            
            # Skip padding (usually w=0, h=0)
            if w_norm <= 0 or h_norm <= 0:
                continue
            
            # 转换回像素尺寸
            w_px = w_norm * VIS_SIZE
            h_px = h_norm * VIS_SIZE
            
            # 收集统计数据 (边长)
            all_patch_widths.append(w_px)

            # 仅可视化第一张图片 (sample_idx = 0)
            if b == sample_idx:
                # Get patch data (1, 8, 8) -> (8, 8)
                patch_tensor = pixel_values[b, i, 0] 
                patch_img = (patch_tensor.numpy() * 255).astype(np.uint8)
                
                # Convert to visualization canvas coords
                x1 = int(x_norm * VIS_SIZE)
                y1 = int(y_norm * VIS_SIZE)
                w = int(max(1, w_norm * VIS_SIZE))
                h = int(max(1, h_norm * VIS_SIZE))
                x2 = x1 + w
                y2 = y1 + h
                
                # -- Reconstruct Input --
                patch_upscaled = cv2.resize(patch_img, (w, h), interpolation=cv2.INTER_NEAREST)
                
                h_patch, w_patch = patch_upscaled.shape
                target_h = min(h_patch, VIS_SIZE - y1)
                target_w = min(w_patch, VIS_SIZE - x1)
                
                if target_h > 0 and target_w > 0:
                    reconstructed_input[y1:y1+target_h, x1:x1+target_w] = patch_upscaled[:target_h, :target_w]
                    
                # -- Visualize Mask --
                is_noised = masks[b, i].item() == 1
                if is_noised:
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 0, 255), 10) # 8K图线条要粗一点 (10px)
                else:
                    cv2.rectangle(mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # ---------------------------
    # Print Statistics
    # ---------------------------
    print("\n" + "="*30)
    print("PATCH SIZE STATISTICS (in pixels)")
    print("="*30)
    if len(all_patch_widths) > 0:
        avg_w = np.mean(all_patch_widths)
        std_w = np.std(all_patch_widths)
        min_w = np.min(all_patch_widths)
        max_w = np.max(all_patch_widths)
        
        print(f"Total patches analyzed: {len(all_patch_widths)}")
        print(f"Average Patch Size: {avg_w:.2f} x {avg_w:.2f} px")
        print(f"Std Dev: {std_w:.2f} px")
        print(f"Smallest Patch: {min_w:.2f} px")
        print(f"Largest Patch:  {max_w:.2f} px")
        
        # 分位数统计
        print("-" * 20)
        print("Percentiles:")
        print(f"25%: {np.percentile(all_patch_widths, 25):.2f} px")
        print(f"50% (Median): {np.median(all_patch_widths):.2f} px")
        print(f"75%: {np.percentile(all_patch_widths, 75):.2f} px")
    else:
        print("No valid patches found in this batch.")
    print("="*30 + "\n")

    # Save results
    print(f"Saving visualization images for {os.path.basename(orig_path)}...")
    cv2.imwrite("vis_original.png", original_img)
    cv2.imwrite("vis_edges.png", edges_vis) 
    cv2.imwrite("vis_model_input.png", reconstructed_input)
    cv2.imwrite("vis_mask_overlay.png", mask_vis)
    
    print("Done! Check vis_original.png, vis_edges.png, vis_model_input.png, and vis_mask_overlay.png")