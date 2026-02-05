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
    负责 HMAE 任务的 Quadtree 分解。
    现在它仅负责生成空间顺序的干净 Patch 序列，具体掩码逻辑移至模型内部。
    """
    def __init__(self, fixed_length=1024, patch_size=32):
        self.fixed_length = fixed_length
        self.patch_size = patch_size

    def process_single(self, img_np, edge_np):
        """
        对单张图像进行处理，返回完整的空间序列补丁。
        """
        h, w, c = img_np.shape
        root = Rect(0, w, 0, h)
        
        active_nodes = [[root.contains(edge_np), root, 0]]
        final_leaves = []

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
                active_nodes.append([child.contains(edge_np), child, depth + 1])
            active_nodes.remove([score, bbox, depth])

        all_leaves = final_leaves + [[n, d] for _, n, d in active_nodes]
        # 严格保持空间顺序
        all_leaves.sort(key=lambda x: (x[0].y1, x[0].x1))

        patches, coords, depths = [], [], []
        
        def to_tensor(p):
            if p.ndim == 2: p = p[..., np.newaxis]
            return torch.from_numpy(p).float().permute(2, 0, 1) / 255.0

        for i, (bbox, depth_val) in enumerate(all_leaves):
            if i >= self.fixed_length: break
            x1, x2, y1, y2 = bbox.get_coord()
            clean_patch = img_np[y1:y2, x1:x2, :]
            
            # Resize 补丁
            clean_patch_resized = cv2.resize(clean_patch, (self.patch_size, self.patch_size))
            
            patches.append(to_tensor(clean_patch_resized))
            coords.append(torch.tensor([x1, x2, y1, y2], dtype=torch.float32))
            depths.append(float(depth_val))

        # Padding
        while len(patches) < self.fixed_length:
            patches.append(torch.zeros(c, self.patch_size, self.patch_size))
            coords.append(torch.zeros(4))
            depths.append(0.0)

        return torch.stack(patches), torch.stack(coords), torch.tensor(depths)