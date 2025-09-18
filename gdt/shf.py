import numpy as np
import cv2 as cv2
import torch
import random
from matplotlib import pyplot as plt

# class Rect:
#     def __init__(self, x1, x2, y1, y2) -> None:
#         # *q
#         # p*
#         self.x1 = x1
#         self.x2 = x2
#         self.y1 = y1
#         self.y2 = y2
        
#         assert x1<=x2, 'x1 > x2, wrong coordinate.'
#         assert y1<=y2, 'y1 > y2, wrong coordinate.'
    
#     def contains(self, domain):
#         patch = domain[self.y1:self.y2, self.x1:self.x2]
#         return int(np.sum(patch)/255)
    
#     def get_area(self, img):
#         return img[self.y1:self.y2, self.x1:self.x2, :]
    
#     def set_area(self, mask, patch):
#         # import pdb;pdb.set_trace()
        
#         patch_size = self.get_size()
#         # patch = np.resize(patch, patch_size)
#         patch = patch.astype('float32')
#         patch = cv.resize(patch, interpolation=cv.INTER_CUBIC , dsize=patch_size)
#         if len(patch.shape)==2:
#             patch = np.expand_dims(patch, axis=-1)

#         mask[self.y1:self.y2, self.x1:self.x2, :] = patch
#         return mask
    
#     def get_coord(self):
#         return self.x1,self.x2,self.y1,self.y2
    
#     def get_size(self):
#         return self.x2-self.x1, self.y2-self.y1
    
#     def get_center(self):
#         return (self.x2+self.x1)/2, (self.y2+self.y1)/2
    
#     def draw(self, ax, c='grey', lw=0.5, **kwargs):
#         # Create a Rectangle patch
#         import matplotlib.patches as patches
#         rect = patches.Rectangle((self.x1, self.y1), 
#                                  width=self.x2-self.x1, 
#                                  height=self.y2-self.y1, 
#                                  linewidth=lw, edgecolor='w', facecolor='none')
#         ax.add_patch(rect)
    
#     def draw_area(self, ax, c='green', lw=0.5, **kwargs):
#         # Create a Rectangle patch
#         import matplotlib.patches as patches
#         rect = patches.Rectangle((self.x1, self.y1), 
#                                  width=self.x2-self.x1, 
#                                  height=self.y2-self.y1, 
#                                  linewidth=lw, edgecolor='w', facecolor=c)
#         ax.add_patch(rect)
    
#     def draw_rescale(self, ax, c='green', lw=0.5, **kwargs):
#         # Create a Rectangle patch
#         import matplotlib.patches as patches
#         rect = patches.Rectangle((self.x1, self.y1), 
#                                  width=16, 
#                                  height=16, 
#                                  linewidth=lw, edgecolor='w', facecolor=c)
#         ax.add_patch(rect)
    
#     def draw_zorder(self, ax, c='red', lw=0.5, **kwargs):
#         # Create a Rectangle patch
#         import matplotlib.patches as patches
#         rect = patches.Rectangle((self.x1, self.y1), 
#                                  width=16, 
#                                  height=16, 
#                                  linewidth=lw, edgecolor='w', facecolor=c)
#         ax.add_patch(rect)
                    
# class FixedQuadTree:
#     def __init__(self, domain, fixed_length=128, build_from_info=False, meta_info=None) -> None:
#         self.domain = domain
#         self.fixed_length = fixed_length
#         if build_from_info:
#             self.nodes = self.decoder_nodes(meta_info=meta_info)
#         else:
#             self._build_tree()
    
#     def nodes_value(self):
#         meta_value = []
#         for rect,v in self.nodes:
#             size,_ = rect.get_size()
#             meta_value += [[size/8]]
#         return meta_value
    
#     def encode_nodes(self):
#         meta_info = []
#         for rect,v in self.nodes:
#             meta_info += [[rect.x1,rect.x2,rect.y1,rect.y2]]
#         return meta_info
    
#     def decoder_nodes(self, meta_info):
#         nodes = []
#         for info in meta_info:
#             x1,x2,y1,y2 = info
#             n = Rect(x1, x2, y1, y2)
#             v = n.contains(self.domain)
#             nodes +=  [[n,v]] 
#         return nodes
            
#     def _build_tree(self):
    
#         h,w = self.domain.shape
#         assert h>0 and w >0, "Wrong img size."
#         root = Rect(0,w,0,h)
#         self.nodes = [[root, root.contains(self.domain)]]
#         while len(self.nodes)<self.fixed_length:
#             bbox, value = max(self.nodes, key=lambda x:x[1])
#             idx = self.nodes.index([bbox, value])
#             if bbox.get_size()[0] == 2:
#                 break

#             x1,x2,y1,y2 = bbox.get_coord()
#             lt = Rect(x1, int((x1+x2)/2), int((y1+y2)/2), y2)
#             v1 = lt.contains(self.domain)
#             rt = Rect(int((x1+x2)/2), x2, int((y1+y2)/2), y2)
#             v2 = rt.contains(self.domain)
#             lb = Rect(x1, int((x1+x2)/2), y1, int((y1+y2)/2))
#             v3 = lb.contains(self.domain)
#             rb = Rect(int((x1+x2)/2), x2, y1, int((y1+y2)/2))
#             v4 = rb.contains(self.domain)
            
#             self.nodes = self.nodes[:idx] + [[lt,v1], [rt,v2], [lb,v3], [rb,v4]] +  self.nodes[idx+1:]

#             # print([v for _,v in self.nodes])
            
#     def count_patches(self):
#         return len(self.nodes)
    
#     def serialize(self, img, size=(8,8,3)):
        
#         seq_patch = []
#         seq_size = []
#         seq_pos = []
#         for bbox,value in self.nodes:
#             seq_patch.append(bbox.get_area(img))
#             seq_size.append(bbox.get_size()[0])
#             seq_pos.append(bbox.get_center())
            
#         h2,w2,c2 = size
        
#         for i in range(len(seq_patch)):
#             h1, w1, c1 = seq_patch[i].shape
#             assert h1==w1, "Need squared input."
#             seq_patch[i] = cv.resize(seq_patch[i], (h2, w2), interpolation=cv.INTER_NEAREST)
#             # assert seq_patch[i].shape == (h2,w2,c2), "Wrong shape {} get, need {}".format(seq_patch[i].shape, (h2,w2,c2))
#         if len(seq_patch)<self.fixed_length:
#             # import pdb
#             # pdb.set_trace()
#             if c2 > 1:
#                 seq_patch += [np.zeros(shape=(h2,w2,c2))] * (self.fixed_length-len(seq_patch))
#             else:
#                 seq_patch += [np.zeros(shape=(h2,w2))] * (self.fixed_length-len(seq_patch))
#             seq_size += [0]*(self.fixed_length-len(seq_size))
#             seq_pos += [tuple([-1,-1])]*(self.fixed_length-len(seq_pos))
#         elif len(seq_patch)>self.fixed_length:
#             pass
#             # random_drop
#         assert len(seq_patch)==self.fixed_length, "Not equal fixed legnth."
#         assert len(seq_size)==self.fixed_length, "Not equal fixed legnth."
#         return seq_patch, seq_size, seq_pos
    
#     def deserialize(self, seq, patch_size, channel):

#         H,W = self.domain.shape
#         seq = np.reshape(seq, (self.fixed_length, patch_size, patch_size, channel))
#         seq = seq.astype(int)
#         mask = np.zeros(shape=(H, W, channel))
#         print("demask:", mask.shape)
        
#         # import pdb;pdb.set_trace()
#         # mask = np.expand_dims(mask, axis=-1)
#         for idx,(bbox,value) in enumerate(self.nodes):
#             pred_mask = seq[idx, ...]
#             mask = bbox.set_area(mask, pred_mask)
#         return mask
    
#     def draw(self, ax, c='grey', lw=1):
#         for bbox,value in self.nodes:
#             bbox.draw(ax=ax)
    
#     def draw_area(self, ax, c='green', lw=1):
#         for bbox,value in self.nodes:
#             bbox.draw_area(ax=ax, c=c, lw=lw)
            
#     def draw_rescale(self, ax, c='green', lw=1):
#         for bbox,value in self.nodes:
#             bbox.draw_rescale(ax=ax, c=c, lw=lw)
            
#     def draw_zorder(self, ax, c='red', lw=1):
#         xs = []
#         ys = []
#         for bbox,value in self.nodes:
#             x,y = bbox.get_center()
#             xs += [x]
#             ys += [y]
#         ax.plot(xs, ys, color='red', linewidth=1)
        
# class Patchify(torch.nn.Module):
#     def __init__(self, sths=[1,3,5,7], fixed_length=1024, cannys=[50, 100], patch_size=8) -> None:
#         super().__init__()
        
#         self.sths = sths
#         self.fixed_length = fixed_length
#         self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
#         self.patch_size = patch_size
        
#     def forward(self, img, target):  # we assume inputs are always structured like this
#         # Do some transformations. Here, we're just passing though the input
        
#         self.smooth_factor = random.choice(self.sths)
#         c = random.choice(self.cannys)
#         self.canny = [c, c+50]
        
#         grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
#         edges = cv.Canny(grey_img, self.canny[0], self.canny[1])
#         qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
#         seq_img = qdt.serialize(img, size=(self.patch_size,self.patch_size,3))
#         seq_img = np.asarray(seq_img)
#         seq_img = np.reshape(seq_img, [self.patch_size, -1, 3])
        
#         seq_mask = qdt.serialize(target, size=(self.patch_size, self.patch_size, 1))
#         seq_mask = np.asarray(seq_mask)
#         seq_mask = np.reshape(seq_mask, [self.patch_size, -1, 1])

#         return seq_img, seq_mask, qdt

# class ImagePatchify(torch.nn.Module):
#     def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3) -> None:
#         super().__init__()
        
#         self.sths = sths
#         self.fixed_length = fixed_length
#         self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
#         self.patch_size = patch_size
#         self.num_channels = num_channels
        
#     def forward(self, img):  # we assume inputs are always structured like this
#         # Do some transformations. Here, we're just passing though the input
#         self.smooth_factor = random.choice(self.sths)
#         c = random.choice(self.cannys)
#         self.canny = [c, c+50]
#         # self.smooth_factor = 0
#         if self.smooth_factor ==0 :
#             edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
#             # edges = np.random.uniform(low=0,high=1, size=(256,256))
#         else:
#             grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
#             # import pdb;pdb.set_trace()
#             edges = cv.Canny(grey_img, self.canny[0], self.canny[1])

#         qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
#         seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.patch_size, self.patch_size, self.num_channels))
#         seq_size = np.asarray(seq_size)
#         seq_img = np.asarray(seq_img)
#         seq_img = np.reshape(seq_img, [self.patch_size*self.patch_size, -1, self.num_channels])

#         return seq_img, seq_size, seq_pos
    
    
class Rect:
    def __init__(self, x1, x2, y1, y2) -> None:
        self.x1 = int(x1); self.x2 = int(x2); self.y1 = int(y1); self.y2 = int(y2)
        assert self.x1 <= self.x2, 'x1 > x2, wrong coordinate.'
        assert self.y1 <= self.y2, 'y1 > y2, wrong coordinate.'
    
    def contains(self, domain):
        if self.y1 >= self.y2 or self.x1 >= self.x2: return 0
        patch = domain[self.y1:self.y2, self.x1:self.x2]
        return int(np.sum(patch)/255)
    
    def get_area(self, img):
        return img[self.y1:self.y2, self.x1:self.x2, :]
    
    def get_coord(self):
        return self.x1, self.x2, self.y1, self.y2
    
    def get_size(self):
        return self.x2 - self.x1, self.y2 - self.y1
    
    def get_center(self):
        return (self.x2 + self.x1) / 2, (self.y2 + self.y1) / 2
    
    def draw(self, ax, c='cyan', lw=1, **kwargs):
        rect = patches.Rectangle((self.x1, self.y1), 
                                 width=self.x2-self.x1, 
                                 height=self.y2-self.y1, 
                                 linewidth=lw, edgecolor=c, facecolor='none')
        ax.add_patch(rect)

class FixedQuadTree:
    def __init__(self, domain, fixed_length=128) -> None:
        self.domain = domain
        self.fixed_length = fixed_length
        self._build_tree()
            
    def _build_tree(self):
        h, w = self.domain.shape
        assert h > 0 and w > 0, "Wrong img size."
        root = Rect(0, w, 0, h)
        self.nodes = [[root, root.contains(self.domain)]]
        
        while len(self.nodes) < self.fixed_length:
            try:
                bbox, value = max(self.nodes, key=lambda x: x[1])
            except ValueError:
                break # 没有可分裂的节点

            idx = self.nodes.index([bbox, value])
            
            width, height = bbox.get_size()
            if width < 2 or height < 2 or value == 0:
                # 找到下一个最佳分裂点
                temp_nodes = self.nodes[:idx] + self.nodes[idx+1:]
                if not temp_nodes: break
                
                # 为了避免无限循环，我们将当前节点的值设为-1，以便不再选择它
                self.nodes[idx][1] = -1
                continue

            x1, x2, y1, y2 = bbox.get_coord()
            cx = x1 + width // 2
            cy = y1 + height // 2
            
            children_defs = [
                (x1, cx, y1, cy), (cx, x2, y1, cy),
                (x1, cx, cy, y2), (cx, x2, cy, y2)
            ]
            
            new_nodes = []
            for (nx1, nx2, ny1, ny2) in children_defs:
                child_rect = Rect(nx1, nx2, ny1, ny2)
                child_val = child_rect.contains(self.domain)
                new_nodes.append([child_rect, child_val])
            
            self.nodes = self.nodes[:idx] + new_nodes + self.nodes[idx+1:]

    def serialize(self, img, patch_size, num_channels):
        seq_patch = []
        seq_size = []
        seq_pos = []
        
        for bbox, value in self.nodes:
            seq_patch.append(bbox.get_area(img))
            seq_size.append(bbox.get_size()[0])
            seq_pos.append(bbox.get_center())
            
        final_patches = []
        for patch in seq_patch:
            resized = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
            final_patches.append(resized)

        num_generated = len(final_patches)
        if num_generated < self.fixed_length:
            padding_needed = self.fixed_length - num_generated
            pad_patch = np.zeros((patch_size, patch_size, num_channels), dtype=np.uint8)
            final_patches.extend([pad_patch] * padding_needed)
            seq_size.extend([0] * padding_needed)
            seq_pos.extend([(0, 0)] * padding_needed)
            
        return final_patches, seq_size, seq_pos

class ImagePatchify:
    def __init__(self, fixed_length=196, patch_size=16, num_channels=3) -> None:
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        self.sths = [0, 1, 3, 5]
        self.cannys = list(range(50, 151, 10))
        
    def __call__(self, img_np):
        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        smooth_factor = random.choice(self.sths)
        
        if smooth_factor == 0:
            edges = (np.random.uniform(low=0, high=255, size=img_np.shape[:2])).astype(np.uint8)
        else:
            grey_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(grey_img, (smooth_factor, smooth_factor), 0)
            canny_t1 = random.choice(self.cannys)
            edges = cv2.Canny(blurred, canny_t1, canny_t1 * 2)

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        
        seq_patches, seq_sizes, seq_pos = qdt.serialize(
            img_np, self.patch_size, self.num_channels
        )
        # 返回qdt以便于在transform中获取坐标信息
        return seq_patches, seq_sizes, seq_pos, qdt

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# --- Visualization and deserialization helper functions ---
def denormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Denormalizes a tensor for display."""
    mean = torch.tensor(mean)
    std = torch.tensor(std)

    if tensor.ndim == 4: # Batch of patches [N, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(0, 2, 3, 1).numpy() # -> [N, H, W, C]
    elif tensor.ndim == 3: # Single image [C, H, W]
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        tensor = tensor.cpu() * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).numpy() # -> [H, W, C]
    else:
        raise TypeError(f"Unsupported tensor dimension: {tensor.ndim}")


def deserialize_patches(patches_tensor, coords_tensor, image_size):
    """Reconstructs an image from serialized patches."""
    reconstructed_img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    
    # First, denormalize all patches
    denormalized_patches = denormalize(patches_tensor) # This call is now safe
    
    for i in range(patches_tensor.shape[0]):
        patch_np = denormalized_patches[i]
        x1, x2, y1, y2 = coords_tensor[i].numpy()
        
        width, height = x2 - x1, y2 - y1
        if width == 0 or height == 0: continue

        # Resize the patch back to its original size in the quadtree
        resized_patch = cv2.resize(patch_np, (width, height), interpolation=cv2.INTER_CUBIC)
        reconstructed_img[y1:y2, x1:x2, :] = resized_patch
        
    return np.clip(reconstructed_img, 0, 1)