import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import random

class Rect:
    def __init__(self, x1, x2, y1, y2):
        self.x1, self.x2, self.y1, self.y2 = int(x1), int(x2), int(y1), int(y2)

    def contains(self, edge_map):
        if self.y1 >= self.y2 or self.x1 >= self.x2: return 0
        patch = edge_map[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch) / 255)

    def get_size(self):
        return self.x2 - self.x1, self.y2 - self.y1

    def get_coord(self):
        return self.x1, self.x2, self.y1, self.y2

class HierarchicalMaskedAutoEncoder:
    """
    负责 HMAE 任务的核心逻辑：
    1. 四叉树层级分解
    2. 噪声注入与掩码生成
    3. 训练损失计算 (train_step_loss)
    """
    def __init__(self, visible_fraction=0.25, fixed_length=1024, patch_size=32, norm_pix_loss=True):
        self.visible_fraction = visible_fraction
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss

    def process_single(self, img_np, edge_np):
        """
        对单张图像进行处理，返回模型所需的各个张量
        """
        h, w, c = img_np.shape
        noise_canvas = np.zeros_like(img_np, dtype=np.float32)
        root = Rect(0, w, 0, h)
        
        active_nodes = [[root.contains(edge_np), root, 0]]
        final_leaves = []

        # 层次化噪声生成
        target_var = np.mean(img_np.astype(np.float32))
        if target_var > 0:
            noise_canvas += np.random.normal(0, np.sqrt(target_var), img_np.shape)

        while len(active_nodes) + len(final_leaves) < self.fixed_length:
            if not active_nodes: break
            score, bbox, depth = max(active_nodes, key=lambda x: x[0])
            if bbox.get_size()[0] <= 4 or score == 0:
                final_leaves.append([bbox, depth])
                active_nodes.remove([score, bbox, depth])
                continue

            x1, x2, y1, y2 = bbox.get_coord()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            children = [Rect(x1, cx, y1, cy), Rect(cx, x2, y1, cy), Rect(x1, cx, cy, y2), Rect(cx, x2, cy, y2)]
            
            for child in children:
                cx1, cx2, cy1, cy2 = child.get_coord()
                curr_var = np.var(noise_canvas[cy1:cy2, cx1:cx2, :])
                target_var = np.mean(img_np[cy1:cy2, cx1:cx2, :].astype(np.float32))
                new_var = max(0, target_var - curr_var)
                if new_var > 0:
                    noise_canvas[cy1:cy2, cx1:cx2, :] += np.random.normal(0, np.sqrt(new_var), (cy2-cy1, cx2-cx1, c))
                active_nodes.append([child.contains(edge_np), child, depth + 1])
            active_nodes.remove([score, bbox, depth])

        all_leaves = final_leaves + [[n, d] for _, n, d in active_nodes]
        all_leaves.sort(key=lambda x: (x[0].y1, x[0].x1))

        indices = list(range(len(all_leaves)))
        random.shuffle(indices)
        visible_set = set(indices[:int(len(all_leaves) * self.visible_fraction)])

        patches, targets, noises, coords, depths, masks = [], [], [], [], [], []
        
        def to_tensor(p):
            # Ensure the patch has a channel dimension (H, W, C)
            # cv2.resize drops the channel dimension for single-channel inputs
            if p.ndim == 2:
                p = p[..., np.newaxis]
            return torch.from_numpy(p).float().permute(2, 0, 1) / 255.0
        
        for i, (bbox, depth_val) in enumerate(all_leaves):
            if i >= self.fixed_length: break
            x1, x2, y1, y2 = bbox.get_coord()
            clean_patch = img_np[y1:y2, x1:x2, :]
            noise_patch = noise_canvas[y1:y2, x1:x2, :]
            
            if i in visible_set:
                input_patch = clean_patch
                mask_val = 0
            else:
                input_patch = np.clip(clean_patch.astype(np.float32) + noise_patch, 0, 255).astype(np.uint8)
                mask_val = 1
            
            input_patch = cv2.resize(input_patch, (self.patch_size, self.patch_size))
            clean_patch = cv2.resize(clean_patch, (self.patch_size, self.patch_size))
            noise_patch = cv2.resize(noise_patch, (self.patch_size, self.patch_size))
            
            patches.append(to_tensor(input_patch))
            targets.append(to_tensor(clean_patch))
            noises.append(to_tensor(noise_patch))
            coords.append(torch.tensor([x1, x2, y1, y2], dtype=torch.float32))
            depths.append(float(depth_val))
            masks.append(mask_val)

        while len(patches) < self.fixed_length:
            patches.append(torch.zeros(c, self.patch_size, self.patch_size))
            targets.append(torch.zeros(c, self.patch_size, self.patch_size))
            noises.append(torch.zeros(c, self.patch_size, self.patch_size))
            coords.append(torch.zeros(4))
            depths.append(0.0)
            masks.append(-1)

        return torch.stack(patches), torch.stack(targets), torch.stack(noises), \
               torch.stack(coords), torch.tensor(depths), torch.tensor(masks)

    def train_step_loss(self, target_patches, pred_patches, target_noise, pred_noise, mask):
        """
        计算训练步骤的 Loss
        """
        # 图像重建损失 (针对被掩码/污染的区域)
        target_img = target_patches.flatten(2)
        if self.norm_pix_loss:
            mean = target_img.mean(dim=-1, keepdim=True)
            var = target_img.var(dim=-1, keepdim=True)
            target_img = (target_img - mean) / (var.sqrt() + 1e-6)
        
        loss_img = (pred_patches - target_img) ** 2
        loss_img = loss_img.mean(dim=-1)

        # 噪声预测损失
        target_n = target_noise.flatten(2)
        if self.norm_pix_loss:
            mean_n = target_n.mean(dim=-1, keepdim=True)
            var_n = target_n.var(dim=-1, keepdim=True)
            target_n = (target_n - mean_n) / (var_n.sqrt() + 1e-6)
        
        loss_n = (pred_noise - target_n) ** 2
        loss_n = loss_n.mean(dim=-1)

        # 总损失
        loss = loss_img + loss_n
        mask_bool = (mask == 1).float()
        mask_sum = mask_bool.sum()
        if mask_sum == 0: return (loss * 0).sum()
        
        return (loss * mask_bool).sum() / mask_sum
        
# # --- Example Usage ---
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Hierarchical HMAE Visualization Demo')
#     parser.add_argument('--image_path', type=str, default="/Users/zhangenzhi/Desktop/zez/Me_and_My_Research/hmae/rescaled_image_0_4096x4096.png", help='Path to a local image to process.')
#     # parser.add_argument('--image_path', type=str, default="/Users/zhangenzhi/Desktop/zez/Me_and_My_Research/hmae/hydrogel_16_1024x1024.jpg", help='Path to a local image to process.')
#     # parser.add_argument('--image_path', type=str, default="/Users/zhangenzhi/Desktop/zez/Me_and_My_Research/hmae/s8d_1_8k_c.png", help='Path to a local image to process.')
#     args = parser.parse_args()
            
#     # --- Start of Demo ---
#     IMG_SIZE = 1024
#     FIXED_LENGTH = 8194
    
#     # Load image from path or create a synthetic one
#     if args.image_path:
#         original_image = cv2.imread(args.image_path)
#         if original_image is None:
#             print(f"Error: Could not load image from {args.image_path}. Using synthetic image.")
#             # Fallback to synthetic image
#             original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
#             original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
#             original_image = cv2.circle(original_image, (128, 128), 90, (200, 150, 100), -1)
#         else:
#             print(f"Loaded image from {args.image_path}")
#             original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE))
#     else:
#         print("No image path provided. Using synthetic image.")
#         original_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
#         original_image = cv2.rectangle(original_image, (0,0), (IMG_SIZE,IMG_SIZE), (30, 30, 30), -1)
#         original_image = cv2.circle(original_image, (128, 128), 90, (200, 150, 100), -1)
#         original_image = cv2.circle(original_image, (90, 100), 20, (20, 40, 60), -1)
    
#     # # hydrogel
#     # gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     # smooth_factor=5
#     # blurred = cv2.GaussianBlur(gray_img, (smooth_factor, smooth_factor), 0)
#     # edges = cv2.Canny(blurred, 120, 170)
    
#     # # s8d
#     # gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     # smooth_factor=5
#     # blurred = cv2.GaussianBlur(gray_img, (smooth_factor, smooth_factor), 0)
#     # edges = cv2.Canny(blurred, 180, 250)
    
#     # paip
#     smooth_factor=3
#     gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray_img, (smooth_factor, smooth_factor), 0)
#     edges = cv2.Canny(blurred, 110, 170)

#     print("Processing with NEW HierarchicalMAEProcessor...")
#     hierarchical_processor = HierarchicalMAEProcessor(visible_fraction=0.25)
#     final_sequence, leaf_nodes, noised_mask, noise_canvas, leaf_depths = hierarchical_processor.create_training_sequence(
#         original_image, edges, fixed_length=FIXED_LENGTH
#     )
    
#     print(f"Generated a sequence of {len(final_sequence)} patches.")
#     print(f"Number of visible patches: {np.sum(noised_mask == 0)}")
#     print(f"Number of noised patches: {np.sum(noised_mask == 1)}")

#     # --- Visualize the result ---
#     # Create the canvas for the HDE input visualization
#     hde_input_canvas = np.zeros_like(original_image, dtype=np.float32)

#     for i, bbox in enumerate(leaf_nodes):
#         if i >= len(noised_mask): break
#         x1, x2, y1, y2 = bbox.get_coord()
#         patch_size = bbox.get_size()
#         if patch_size[0] == 0 or patch_size[1] == 0: continue

#         is_noised = noised_mask[i] == 1
#         if is_noised:
#             noise_patch = noise_canvas[y1:y2, x1:x2, :]
#             noise_vis = noise_patch - np.mean(noise_patch)
#             if np.std(noise_vis) > 1e-6:
#                 noise_vis = noise_vis / (np.std(noise_vis) * 3)
#             noise_vis = (noise_vis + 0.5).clip(0, 1)
#             hde_input_canvas[y1:y2, x1:x2, :] = noise_vis
#         else:
#             original_patch = bbox.get_area(original_image)
#             hde_input_canvas[y1:y2, x1:x2, :] = original_patch.astype(np.float32) / 255.0

#     if hde_input_canvas.shape[2] == 3:
#         hde_input_canvas_rgb = cv2.cvtColor((hde_input_canvas * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0
#     else:
#         hde_input_canvas_rgb = hde_input_canvas


#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
#     # Panel 1: 原始图像 (Original Image)
#     axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
#     axes[0, 0].set_title('Original Image')
#     axes[0, 0].axis('off')
    
#     # Panel 2: 边缘图与自适应网格 (Edge Map + Quadtree Grid)
#     axes[0, 1].imshow(edges, cmap='gray')
#     axes[0, 1].set_title('Edge Map + Quadtree')
#     for bbox in leaf_nodes:
#         rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
#                              linewidth=0.8, edgecolor='cyan', facecolor='none')
#         axes[0, 1].add_patch(rect)
#     axes[0, 1].axis('off')
    
#     # Panel 3: HDE 模型输入 (HDE Input)
#     axes[1, 0].imshow(hde_input_canvas_rgb)
#     axes[1, 0].set_title('DE Input: Clean=G, Noised=R')
#     for i, bbox in enumerate(leaf_nodes):
#          if i >= len(noised_mask): break
#          is_noised = noised_mask[i] == 1
#          edge_color = 'r' if is_noised else 'g'
#          rect = plt.Rectangle((bbox.x1, bbox.y1), bbox.get_size()[0], bbox.get_size()[1], 
#                               linewidth=0.8, edgecolor=edge_color, facecolor='none')
#          axes[1, 0].add_patch(rect)
#     axes[1, 0].axis('off')
    
#     # Panel 4: 噪声结构图 (Patch-level Noise Variance)
#     patch_variance_canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
#     for i, bbox in enumerate(leaf_nodes):
#         x1, x2, y1, y2 = bbox.get_coord()
        
#         # Extract the noise from the corresponding patch
#         noise_patch = noise_canvas[y1:y2, x1:x2, :]
        
#         # Calculate the variance within that patch
#         patch_variance = np.var(noise_patch)
        
#         # Fill the canvas with this variance value
#         patch_variance_canvas[y1:y2, x1:x2] = patch_variance

#     # Normalize the entire canvas for visualization
#     min_var, max_var = np.min(patch_variance_canvas), np.max(patch_variance_canvas)
#     if max_var > min_var:
#         patch_variance_canvas_vis = (patch_variance_canvas - min_var) / (max_var - min_var)
#     else:
#         patch_variance_canvas_vis = np.zeros_like(patch_variance_canvas)
    
#     im = axes[1, 1].imshow(patch_variance_canvas_vis, cmap='inferno')
#     axes[1, 1].set_title('Noise Structure')
#     axes[1, 1].axis('off')
#     fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.show()
