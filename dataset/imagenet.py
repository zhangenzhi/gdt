import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import os
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm

def imagenet(args):

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    image_datasets = {x: datasets.ImageNet(root=args.data_dir, split=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
    
    # Convert datasets to InMemoryDataset
    # in_memory_datasets = {x: InMemoryDataset(image_datasets[x]) for x in ['train', 'val']}


    # Create data loaders
    shuffle = True
    pin_memory  = True
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=shuffle, 
                                 num_workers=args.num_workers, pin_memory=pin_memory)
                   for x in ['train', 'val']}
    return dataloaders

from .transform import ImagenetTransformArgs, build_transform
# 建议将 num_workers 作为参数传入，而不是依赖外部的 args
def imagenet_distribute(img_size, data_dir, batch_size, num_workers=32):
    """
    为分布式训练优化的ImageNet DataLoader函数
    """
    # 数据增强部分保持不变
    data_transforms = {
        'train':build_transform(is_train=True, args=ImagenetTransformArgs(input_size=224)),
        'val': build_transform(is_train=False, args=ImagenetTransformArgs(input_size=224))
    }

    # 创建数据集
    image_datasets = {x: datasets.ImageNet(root=data_dir, split=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
                      
    # 为训练集和验证集创建分布式采样器
    samplers = {x: DistributedSampler(image_datasets[x], shuffle=True) for x in ['train', 'val']}
    # 注意：DistributedSampler 默认会打乱数据，所以DataLoader的shuffle参数必须为False。

    # 创建优化后的DataLoaders
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            num_workers=num_workers,      # 优化点 1: 启用多进程加载
            pin_memory=True,              # 优化点 2: 加速CPU到GPU的数据传输
            sampler=samplers['train'],
            drop_last=True                # 优化点 3: 避免分布式训练中的同步问题
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=samplers['val']
            # 验证集通常不需要 drop_last=True
        )
    }
    
    return dataloaders

def build_mae_dataloaders(img_size, data_dir, batch_size, num_workers=32):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # MAE pre-training data augmentation
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Validation data augmentation
    transform_val = transforms.Compose([
        transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # 数据增强部分保持不变
    data_transforms = {'train':transform_train, 'val': transform_val}

    # 创建数据集
    image_datasets = {x: datasets.ImageNet(root=data_dir, split=x, transform=data_transforms[x]) for x in ['train', 'val']}
    samplers = {x: DistributedSampler(image_datasets[x], shuffle=True) for x in ['train', 'val']}
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            num_workers=num_workers,     
            pin_memory=True,              
            sampler=samplers['train'],
            drop_last=True                
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=samplers['val']
        )
    }
    
    return dataloaders


from gdt.hde import HierarchicalHDEProcessor, Rect

# --- 辅助函数：将 PyTorch 张量转换为 OpenCV 图像 ---

def tensor_to_cv2_img(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将一个PyTorch张量（归一化，CHW）转换为一个NumPy数组（HWC，uint8），
    以便OpenCV使用。
    """
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    tensor = inv_normalize(tensor)
    numpy_img = tensor.permute(1, 2, 0).numpy()
    numpy_img = (numpy_img * 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    return bgr_img

# --- 用于HDE的自定义PyTorch数据集 ---

class HDEImageNetDataset(Dataset):
    """
    一个自定义的PyTorch数据集，它包装了ImageNet数据集。
    对于每张图像，它都会执行完整的HDE预处理流程，使用最新的HierarchicalHDEProcessor。
    """
    def __init__(self, underlying_dataset, fixed_length=256, patch_size=16, visible_fraction=0.25):
        self.underlying_dataset = underlying_dataset
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.hde_processor = HierarchicalHDEProcessor(visible_fraction=visible_fraction)
        
        # 用于随机化Canny边缘检测的参数
        self.canny_thresholds = list(range(50, 151, 10))

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        # 1. 从基础数据集中获取原始的、经过变换的图像张量
        img_tensor, target_label = self.underlying_dataset[idx]
        
        # 2. 将张量转换回与CV2兼容的NumPy图像
        img_np = tensor_to_cv2_img(img_tensor)

        # 3. 执行边缘检测
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        t1 = random.choice(self.canny_thresholds)
        t2 = t1 * 2
        edges = cv2.Canny(gray_img, t1, t2)
        
        # 4. 使用新的分层处理器生成图像块序列和元数据
        patch_sequence, leaf_nodes, noised_mask, _, leaf_depths = \
            self.hde_processor.create_training_sequence(img_np, edges, self.fixed_length)
        
        # 5. 为模型序列化输出
        num_channels = img_np.shape[2]
        
        # 预分配数组
        final_patches = np.zeros((self.fixed_length, self.patch_size, self.patch_size, num_channels), dtype=np.float32)
        seq_coords = np.zeros((self.fixed_length, 4), dtype=np.int32)
        seq_depths = np.zeros(self.fixed_length, dtype=np.int32)

        for i, patch in enumerate(patch_sequence):
            # 将图像块大小调整为ViT期望的固定大小
            resized_patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            final_patches[i] = resized_patch.astype(np.float32)
            
            # 获取元数据
            bbox = leaf_nodes[i]
            seq_coords[i] = bbox.get_coord()
            seq_depths[i] = leaf_depths[i]
            
        # 6. 将所有内容转换为PyTorch张量
        # 将图像块从[0, 255]归一化到[0, 1]，并从HWC转换为CHW
        patches_tensor = torch.from_numpy(final_patches).permute(0, 3, 1, 2) / 255.0
        
        # 应用标准的归一化
        normalize_output = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        patches_tensor = normalize_output(patches_tensor)

        return {
            "patches": patches_tensor,
            "coords": torch.from_numpy(seq_coords),
            "depths": torch.from_numpy(seq_depths),
            "mask": torch.from_numpy(noised_mask),
            "target": target_label,
            "original_image": img_tensor # 如果需要，返回以用于可视化/调试
        }

# --- HDE Dataloader构建函数 ---

def build_hde_imagenet_dataloaders(img_size, data_dir, batch_size, fixed_length=256, patch_size=16, num_workers=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 基础图像的标准数据增强
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print("正在加载基础ImageNet数据集...")
    base_image_datasets = {x: datasets.ImageNet(root=data_dir, split=x, transform=transform_train if x == 'train' else transform_val) for x in ['train', 'val']}
    print("基础数据集加载完成。")
    
    print("正在使用HDEImageNetDataset进行包装...")
    hde_datasets = {x: HDEImageNetDataset(base_image_datasets[x], fixed_length=fixed_length, patch_size=patch_size) for x in ['train', 'val']}
    print("HDE数据集创建完成。")

    if dist.is_available() and dist.is_initialized():
        print("正在使用DistributedSampler。")
        samplers = {x: DistributedSampler(hde_datasets[x], shuffle=True) for x in ['train', 'val']}
    else:
        print("正在使用标准的RandomSampler/SequentialSampler。")
        samplers = {
            'train': torch.utils.data.RandomSampler(hde_datasets['train']),
            'val': torch.utils.data.SequentialSampler(hde_datasets['val'])
        }

    dataloaders = {
        'train': DataLoader(
            hde_datasets['train'],
            batch_size=batch_size,
            num_workers=num_workers,     
            pin_memory=True,              
            sampler=samplers['train'],
            drop_last=True                
        ),
        'val': DataLoader(
            hde_datasets['val'],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=samplers['val']
        )
    }
    
    return dataloaders

# --- 可视化函数 ---
def visualize_batch(batch, save_path="hde_dataloader_visualization.png"):
    # 此函数用于调试和验证，它可视化一个批次中的前几个样本
    print(f"正在生成可视化图像并保存至 {save_path}...")
    # ... 实现与 hde_utils.py 中类似的详细可视化逻辑 ...
    # 为了简洁，这里只打印形状信息
    print("批次键和形状:")
    for key, value in batch.items():
        print(f"  - {key}: {value.shape}")


from gdt.shf import ImagePatchify
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# --- 集成了timm数据增强的自定义Transform类 ---
class SHFQuadtreeTransform:
    def __init__(self, is_train, transform_args, fixed_length=196, patch_size=16):
        """
        初始化Transform。
        - is_train: 布尔值，指示是否为训练集。
        - transform_args: 包含timm create_transform所需参数的对象。
        """
        # 这个base_transform只包含作用于PIL图像的增强
        self.base_transform = self._build_pil_transform(is_train, transform_args)
        
        self.patchify = ImagePatchify(fixed_length=fixed_length, patch_size=patch_size, num_channels=3)

    def _build_pil_transform(self, is_train, args):
        """构建只包含PIL操作的数据增强部分。"""
        if is_train:
            # 使用timm的create_transform来获取标准的增强策略
            timm_transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
            # timm的transform默认最后两步是ToTensor和Normalize。
            # 我们的流程需要在中间插入numpy操作，所以只取前面的PIL增强。
            return transforms.Compose(timm_transform.transforms[:-2])
        else:
            # 为验证集构建标准的Resize和CenterCrop流程
            if args.input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(args.input_size / crop_pct)
            return transforms.Compose([
                transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(args.input_size),
            ])

    def __call__(self, pil_img):
        # 1. 应用基础的PIL数据增强
        augmented_pil = self.base_transform(pil_img)
        
        # 2. 将PIL图像转换为NumPy数组以进行cv2处理 (RGB -> BGR)
        img_np = cv2.cvtColor(np.array(augmented_pil), cv2.COLOR_RGB2BGR)
        
        # 3. 应用自定义的ImagePatchify逻辑
        seq_patches, seq_sizes, seq_pos = self.patchify(img_np)
        
        # 4. 将结果转换为Tensors
        patches_np = np.stack(seq_patches, axis=0)
        patches_tensor = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float()

        sizes_tensor = torch.tensor(seq_sizes, dtype=torch.long)
        positions_tensor = torch.tensor(seq_pos, dtype=torch.float32)
        
        # 5. 手动进行归一化，这是此流程的正确步骤
        normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        patches_tensor_normalized = normalize(patches_tensor / 255.0)

        return {
            "patches": patches_tensor_normalized,
            "sizes": sizes_tensor,
            "positions": positions_tensor
        }

# --- 主Dataloader构建函数 ---
def build_shf_imagenet_dataloader(img_size, data_dir, batch_size, num_workers=32):
    fixed_length = (img_size // 16) ** 2 # 假设patch大小为16
    patch_size = 16

    # 为训练和验证创建timm数据增强的参数
    train_transform_args = ImagenetTransformArgs(input_size=img_size)
    val_transform_args = ImagenetTransformArgs(input_size=img_size)

    # 创建集成了timm的自定义Transform
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
            pin_memory=True, sampler=samplers['train'], drop_last=True
        ),
        'val': DataLoader(
            image_datasets['val'], batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, sampler=samplers['val']
        )
    }
    
    return dataloaders

def create_imagenet_subset(original_dir, subset_dir, num_classes, imgs_per_class):
    """
    Creates a subset of the ImageNet dataset.

    Args:
        original_dir (str): Path to the original ImageNet directory (e.g., '.../train').
        subset_dir (str): Path to save the subset.
        num_classes (int): Number of classes to randomly select.
        imgs_per_class (int): Number of images to randomly select from each class.
    """
    # Create the subset directory if it doesn't exist
    if os.path.exists(subset_dir):
        print(f"Directory '{subset_dir}' already exists. Deleting it to create a fresh subset.")
        shutil.rmtree(subset_dir)
    os.makedirs(subset_dir)
    print(f"Created new directory: '{subset_dir}'")

    # List all the classes (folders) in the original dataset directory
    classes = [d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))]
    
    if not classes:
        print(f"Error: No class folders found in '{original_dir}'. Please check the path.")
        return

    # Randomly select classes for the subset
    print(f"Selecting {num_classes} random classes from {len(classes)} total classes...")
    selected_classes = random.sample(classes, k=min(num_classes, len(classes)))

    # Iterate through the selected classes with a progress bar
    for class_name in tqdm(selected_classes, desc="Processing classes"):
        # Create the class directory in the subset directory
        subset_class_dir = os.path.join(subset_dir, class_name)
        os.makedirs(subset_class_dir)

        original_class_dir = os.path.join(original_dir, class_name)
        class_images = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]

        # Randomly select images for the subset
        # Ensure we don't try to sample more images than available
        k_images = min(imgs_per_class, len(class_images))
        selected_images = random.sample(class_images, k=k_images)

        # Copy the selected images to the subset directory
        for image_name in selected_images:
            original_image_path = os.path.join(original_class_dir, image_name)
            subset_image_path = os.path.join(subset_class_dir, image_name)
            shutil.copy(original_image_path, subset_image_path)

    print(f"\nSubset creation for '{original_dir}' completed.")
    print(f"Total classes: {len(selected_classes)}")
    print(f"Images per class: up to {imgs_per_class}")

def imagenet_subloaders(subset_data_dir, batch_size=32, num_workers=4):
    """
    Creates PyTorch DataLoaders for the ImageNet subset.

    Args:
        subset_data_dir (str): Path to the root of the subset data (containing 'train' and 'val').
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple: A tuple containing (dataloaders, class_names).
               'dataloaders' is a dictionary {'train': dataloader, 'val': dataloader}.
               'class_names' is a list of the class names.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Define transformations for training (with data augmentation) and validation (no augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ]),
    }

    # Create ImageFolder datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(subset_data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # Create DataLoader objects
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ['train', 'val']
    }

    # Get class names
    class_names = image_datasets['train'].classes
    
    print("DataLoaders created successfully.")
    print(f"Training set has {len(image_datasets['train'])} images.")
    print(f"Validation set has {len(image_datasets['val'])} images.")
    print(f"Number of classes: {len(class_names)}")

    return dataloaders, class_names
        
# epoch iteration
def imagenet_iter(args):
    dataloaders = imagenet(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    for e in range(args.num_epochs):
        start_time = time.time()
        for phase in ['train', 'val']:
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                if step%1000==0:
                    print(step)
        print("Time cost for loading {}".format(time.time() - start_time))
           

# if __name__ == "__main__":
#     # 假设 args 已经通过某个函数（如 parse_args()）正确设置
#     parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
#     parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
#     # parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
#     parser.add_argument('--data_dir', type=str, default='/work/c30636/dataset/imagenet/', help='Path to the ImageNet dataset directory')
#     parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
#     parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for DataLoader')
#     parser.add_argument('--num_workers', type=int, default=48, help='Number of workers for DataLoader')
    
#     args = parser.parse_args()
    
#     dataloaders = imagenet(args)
    
#     import time
#     start_time = time.time()
#     for phase in ['train', 'val']:
#         for step, (inputs, labels) in enumerate(dataloaders[phase]):
#             if step % 100 == 0:
#                 # 为了验证，可以打印出变量的类型和形状
#                 print(f"Phase: {phase}, Step: {step}, Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

#     print("Time cost for loading {}".format(time.time() - start_time))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Create a subset of ImageNet for testing.")
#     # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet", help="Path to the original ImageNet directory containing 'train' and 'val' folders.")
#     # parser.add_argument('--output_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help="Path to save the generated subset.")
#     parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help="Path to the original ImageNet directory containing 'train' and 'val' folders.")
#     parser.add_argument('--output_dir', type=str, default="/work/c30636/gdt/dataset", help="Path to save the generated subset.")
#     parser.add_argument('--num_classes', type=int, default=10, help="Number of classes to include in the subset.")
#     parser.add_argument('--train_imgs', type=int, default=500, help="Number of training images per class.")
#     parser.add_argument('--val_imgs', type=int, default=100, help="Number of validation images per class.")
    
#     args = parser.parse_args()

#     # --- Create Training Subset ---
#     print("--- Starting Training Subset Creation ---")
#     original_train_dir = os.path.join(args.data_dir, 'train')
#     subset_train_dir = os.path.join(args.output_dir, 'train')
#     create_imagenet_subset(original_train_dir, subset_train_dir, args.num_classes, args.train_imgs)

#     # --- Create Validation Subset from the same classes ---
#     print("\n--- Starting Validation Subset Creation ---")
#     # Get the classes selected for the training set to ensure consistency
#     selected_train_classes = os.listdir(subset_train_dir)
#     original_val_dir = os.path.join(args.data_dir, 'val')
#     subset_val_dir = os.path.join(args.output_dir, 'val')
    
#     if os.path.exists(subset_val_dir):
#         shutil.rmtree(subset_val_dir)
#     os.makedirs(subset_val_dir)
    
#     for class_name in tqdm(selected_train_classes, desc="Processing validation classes"):
#         subset_class_dir = os.path.join(subset_val_dir, class_name)
#         os.makedirs(subset_class_dir)
        
#         original_class_dir = os.path.join(original_val_dir, class_name)
#         if not os.path.isdir(original_class_dir):
#             # print(f"Warning: Class '{class_name}' not found in validation set. Skipping.")
#             continue
            
#         class_images = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]
#         k_images = min(args.val_imgs, len(class_images))
#         if k_images == 0:
#             continue
        
#         selected_images = random.sample(class_images, k=k_images)
        
#         for image_name in selected_images:
#             original_image_path = os.path.join(original_class_dir, image_name)
#             subset_image_path = os.path.join(subset_class_dir, image_name)
#             shutil.copy(original_image_path, subset_image_path)
            
#     print("\nValidation subset creation completed.")
#     print("Subset creation finished. Find your data at:", args.output_dir)
    
# if __name__ == '__main__':
#     # --- Example Usage ---
#     # This should be the same path you used for --output_dir in the previous script
#     SUBSET_PATH = "/lustre/orion/nro108/world-shared/enzhi/gdt/dataset"
    
#     if not os.path.exists(os.path.join(SUBSET_PATH, 'train')):
#         print(f"Error: Subset path '{SUBSET_PATH}' does not seem to contain 'train' directory.")
#         print("Please run the create_subset.py script first.")
#     else:
#         # Get the dataloaders
#         dataloaders, class_names = imagenet_subloaders(subset_data_dir=SUBSET_PATH, batch_size=64)

#         # --- Test the dataloader by fetching one batch ---
#         print("\n--- Testing the 'train' dataloader ---")
#         inputs, classes = next(iter(dataloaders['train']))

#         print(f"Shape of one batch of inputs (images): {inputs.shape}")
#         print(f"Shape of one batch of classes (labels): {classes.shape}")
#         print(f"Example labels: {classes[:5]}")
#         # The labels are indices. You can map them back to class names (folder names)
#         print(f"Corresponding class names: {[class_names[i] for i in classes[:5]]}")

# if __name__ == '__main__':
#     import time
#     parser = argparse.ArgumentParser(description='PyTorch HDE ImageNet DataLoader 测试脚本')
#     parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='ImageNet数据集的路径')
#     parser.add_argument('--num_epochs', type=int, default=1, help='迭代的轮数')
#     parser.add_argument('--batch_size', type=int, default=512, help='DataLoader的批次大小')
#     parser.add_argument('--num_workers', type=int, default=32, help='DataLoader的工作线程数')
#     parser.add_argument('--img_size', type=int, default=256, help='图像大小')
#     parser.add_argument('--fixed_length', type=int, default=196, help='图像块数量 (序列长度)')
#     parser.add_argument('--patch_size', type=int, default=16, help='每个图像块调整后的大小')
#     parser.add_argument('--visualize', action='store_true', help='生成并保存一个批次的可视化结果')
    
#     args = parser.parse_args()
    
#     dataloaders = build_hde_imagenet_dataloaders(
#         img_size=args.img_size,
#         data_dir=args.data_dir,
#         batch_size=args.batch_size,
#         fixed_length=args.fixed_length,
#         patch_size=args.patch_size,
#         num_workers=args.num_workers
#     )
    
#     if args.visualize:
#         try:
#             vis_batch = next(iter(dataloaders['val']))
#             # 在一个真实场景中，您会调用一个更复杂的函数来创建像我们之前那样的四宫格图
#             visualize_batch(vis_batch) 
#         except Exception as e:
#             print(f"无法生成可视化: {e}")

#     print("数据加载器已构建。开始迭代...")
#     start_time = time.time()
    
#     for epoch in range(args.num_epochs):
#         for phase in ['train', 'val']:
#             for step, batch in enumerate(dataloaders[phase]):
#                 if step % 100 == 0:
#                     print(f"阶段: {phase}, 步骤: {step}, 图像块形状: {batch['patches'].shape}, 标签形状: {batch['target'].shape}")
#                 if step > 200: # 快速测试，只迭代几步
#                     break
    
#     total_time = time.time() - start_time
#     print(f"\n加载 {args.num_epochs} 轮次的部分数据所花费的总时间: {total_time:.2f} 秒")
    
    
# --- 使用示例与健全性检查 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHF Quadtree Dataloader with Timm Augmentation Test')
    parser.add_argument('--data_dir', type=str, required=True, help='ImageNet数据集的路径。')
    parser.add_argument('--batch_size', type=int, default=4, help='用于测试的批次大小。')
    parser.add_argument('--num_workers', type=int, default=2, help='工作线程数。')
    args = parser.parse_args()

    dataloaders = build_shf_imagenet_dataloader(
        img_size=224,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n--- Dataloader健全性检查 ---")
    print("正在从训练集中获取一个批次...")
    
    batch = next(iter(dataloaders['train']))
    
    print("批次获取成功！")
    print("批次中的键:", batch.keys())
    
    patches = batch['patches']
    sizes = batch['sizes']
    positions = batch['positions']
    
    print(f"\n'patches'张量的形状: {patches.shape}")
    print(f"'sizes'张量的形状: {sizes.shape}")
    print(f"'positions'张量的形状: {positions.shape}")

    assert patches.shape == (args.batch_size, 196, 3, 16, 16)
    assert sizes.shape == (args.batch_size, 196)
    assert positions.shape == (args.batch_size, 196, 2)
    
    print("\n✅ 所有形状都正确！")
    print("健全性检查通过。")
