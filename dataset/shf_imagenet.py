from gdt.shf import ImagePatchify
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import os
import sys
sys.path.append("../")
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
import PIL
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from  dataset.utlis import ImagenetTransformArgs

# --- Custom Transform class with integrated timm augmentation ---
class SHFQuadtreeTransform:
    def __init__(self, is_train, transform_args, fixed_length=196, patch_size=16):
        self.base_transform = self._build_pil_transform(is_train, transform_args)
        self.patchify = ImagePatchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=3, is_train=is_train)

    def _build_pil_transform(self, is_train, args):
        if is_train:
            timm_transform = create_transform(
                input_size=args.input_size, is_training=True,
                color_jitter=args.color_jitter, auto_augment=args.aa,
                interpolation='bicubic', re_prob=args.reprob,
                re_mode=args.remode, re_count=args.recount,
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
            )
            return transforms.Compose(timm_transform.transforms[:-2])
        else:
            crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
            size = int(args.input_size / crop_pct)
            return transforms.Compose([
                transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.input_size),
            ])

    def __call__(self, pil_img):
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        augmented_pil = self.base_transform(pil_img)

        if augmented_pil.mode != 'RGB':
            augmented_pil = augmented_pil.convert('RGB')
        
        # --- [KEY CHANGE] ---
        # Convert directly to a NumPy array without changing color space. It's now RGB.
        img_np = np.array(augmented_pil)
        seq_patches, seq_sizes, seq_pos, qdt = self.patchify(img_np)

        patches_np = np.stack(seq_patches, axis=0)
        # Permute from [N, P, P, 3(RGB)] to [N, 3(RGB), P, P]
        patches_tensor = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float()

        sizes_tensor = torch.tensor(seq_sizes, dtype=torch.long)
        positions_tensor = torch.tensor(seq_pos, dtype=torch.float32)
        
        # # Get coordinates for visualization
        # coords = [node[0].get_coord() for node in qdt.nodes]
        # if len(coords) < self.patchify.fixed_length:
        #     padding_needed = self.patchify.fixed_length - len(coords)
        #     coords.extend([(0,0,0,0)] * padding_needed)
        # coords_tensor = torch.tensor(coords, dtype=torch.long)

        # Get the original image tensor for visualization
        original_tensor = transforms.ToTensor()(augmented_pil)
        
        # Apply normalization. The input tensor is now RGB, so this is correct.
        normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        patches_tensor_normalized = normalize(patches_tensor / 255.0)
        original_tensor_normalized = normalize(original_tensor)

        return {
            "patches": patches_tensor_normalized,
            "sizes": sizes_tensor,
            "positions": positions_tensor,
            # "coords": coords_tensor, # Added
            "original_image": original_tensor_normalized # Added
        }

# --- 主Dataloader构建函数 ---
def build_shf_imagenet_dataloader(img_size, patch_size, fixed_length, data_dir, batch_size, num_workers=32):
    train_transform_args = ImagenetTransformArgs(input_size=img_size)
    val_transform_args = ImagenetTransformArgs(input_size=img_size)

    data_transforms = {
        'train': SHFQuadtreeTransform(is_train=True, transform_args=train_transform_args, fixed_length=fixed_length, patch_size=patch_size),
        'val': SHFQuadtreeTransform(is_train=False, transform_args=val_transform_args, fixed_length=fixed_length, patch_size=patch_size)
    }

    print("正在使用集成了timm的SHFQuadtreeTransform加载ImageNet数据集...")
    image_datasets = {x: datasets.ImageNet(root=data_dir, split=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
                      
    if dist.is_available() and dist.is_initialized():
        print("正在使用DistributedSampler。")
        samplers = {x: DistributedSampler(image_datasets[x], shuffle=True) for x in ['train', 'val']}
    else:
        print("正在使用标准的RandomSampler/SequentialSampler。")
        samplers = {
            'train': torch.utils.data.RandomSampler(image_datasets['train']),
            'val': torch.utils.data.SequentialSampler(image_datasets['val'])
        }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'], batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, sampler=samplers['train'], drop_last=True,prefetch_factor=1
        ),
        'val': DataLoader(
            image_datasets['val'], batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, sampler=samplers['val'],prefetch_factor=1
        )
    }
    
    return dataloaders
