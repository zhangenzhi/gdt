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

from transform import ImagenetTransformArgs, build_transform
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

import sys
sys.path.append("./")
from gdt.hde import HDEProcessor, Rect, FixedQuadTree

def tensor_to_cv2_img(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Converts a PyTorch tensor (normalized, CHW) to a NumPy array (HWC, uint8)
    that can be used by OpenCV.
    """
    # 1. Reverse the normalization
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    tensor = inv_normalize(tensor)
    
    # 2. Convert from CHW to HWC format
    numpy_img = tensor.permute(1, 2, 0).numpy()
    
    # 3. Denormalize from [0, 1] to [0, 255] and convert to uint8
    numpy_img = (numpy_img * 255).astype(np.uint8)
    
    # 4. Convert RGB to BGR for OpenCV
    bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    
    return bgr_img

# --- Visualization Function ---
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def denormalize_for_plot(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor for matplotlib visualization."""
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    tensor = inv_normalize(tensor.cpu())
    numpy_img = tensor.permute(1, 2, 0).numpy()
    numpy_img = np.clip(numpy_img, 0, 1)  # Clip to valid range for imshow
    return numpy_img

def visualize_batch(batch, save_path="hde_visualization.png"):
    """
    Creates and saves a visualization of the HDE process for a batch of images.
    """
    original_images = batch['original_image']
    hde_patches_batch = batch['patches']
    coords_batch = batch['coords']
    mask_batch = batch['mask']
    
    num_images_to_show = min(4, original_images.shape[0])
    fig, axs = plt.subplots(num_images_to_show, 3, figsize=(15, 5 * num_images_to_show), squeeze=False)
    fig.suptitle("HDE Dataloader Visualization", fontsize=16)

    for i in range(num_images_to_show):
        # --- 1. Original Image ---
        original_img_np = denormalize_for_plot(original_images[i])
        axs[i, 0].imshow(original_img_np)
        axs[i, 0].set_title(f"Image {i+1}: Original")
        axs[i, 0].axis('off')

        # --- 2. Reconstruct HDE Image from Patches ---
        h, w, _ = original_img_np.shape
        reconstructed_img = np.zeros_like(original_img_np)
        
        hde_patches = hde_patches_batch[i]
        coords = coords_batch[i]

        for j in range(hde_patches.shape[0]):  # Iterate through patches
            patch_tensor = hde_patches[j]
            patch_np = denormalize_for_plot(patch_tensor)
            
            x1, x2, y1, y2 = coords[j].numpy()
            original_h, original_w = y2 - y1, x2 - x1

            if original_h == 0 or original_w == 0: continue  # Skip padding patches

            resized_patch = cv2.resize(patch_np, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
            if resized_patch.ndim == 2: # Handle grayscale case if it ever occurs
                resized_patch = np.expand_dims(resized_patch, axis=-1)
            reconstructed_img[y1:y2, x1:x2, :] = resized_patch

        axs[i, 1].imshow(reconstructed_img)
        axs[i, 1].set_title("Reconstructed (Clean + Noised)")
        axs[i, 1].axis('off')

        # --- 3. Quadtree Overlay ---
        axs[i, 2].imshow(reconstructed_img)
        axs[i, 2].set_title("Quadtree Overlay (Clean=G, Noised=R)")
        mask = mask_batch[i]

        for j in range(coords.shape[0]):
            x1, x2, y1, y2 = coords[j].numpy()
            patch_w, patch_h = x2 - x1, y2 - y1
            if patch_w == 0 or patch_h == 0: continue

            is_noised = mask[j].item() == 1
            edge_color = 'r' if is_noised else 'g'
            
            rect = patches.Rectangle((x1, y1), patch_w, patch_h, linewidth=1.2, edgecolor=edge_color, facecolor='none')
            axs[i, 2].add_patch(rect)
        axs[i, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")
    
# --- Custom PyTorch Dataset for HDE ---

class HDEDataset(Dataset):
    """
    A custom PyTorch Dataset that wraps an existing image dataset (e.g., ImageNet).
    For each image, it performs the full HDE preprocessing pipeline.
    """
    def __init__(self, underlying_dataset, fixed_length=512, patch_size=16, visible_fraction=0.25):
        self.underlying_dataset = underlying_dataset
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.hde_processor = HDEProcessor(visible_fraction=visible_fraction)
        
        # For randomizing Canny edge detection
        self.canny_thresholds = list(range(50, 151, 10))

    def __len__(self):
        return len(self.underlying_dataset)

    def __getitem__(self, idx):
        # 1. Get the original, transformed image tensor from the base dataset
        img_tensor, target_label = self.underlying_dataset[idx]
        
        # 2. Convert tensor back to a CV2-compatible NumPy image
        img_np = tensor_to_cv2_img(img_tensor)

        # 3. Perform edge detection
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        t1 = random.choice(self.canny_thresholds)
        t2 = t1 * 2
        edges = cv2.Canny(gray_img, t1, t2)
        
        # 4. Build the adaptive quadtree
        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        
        # 5. Generate the sequence of clean and noised patches
        patch_sequence, noised_mask = self.hde_processor.create_training_sequence(img_np, qdt)
        
        # 6. Serialize the output for the model
        num_channels = img_np.shape[2]
        
        # Pre-allocate arrays
        final_patches = np.zeros((self.fixed_length, self.patch_size, self.patch_size, num_channels), dtype=np.float32)
        seq_size = np.zeros(self.fixed_length, dtype=np.int32)
        seq_pos = np.zeros((self.fixed_length, 2), dtype=np.float32)
        seq_coords = np.zeros((self.fixed_length, 4), dtype=np.int32)

        leaf_nodes = [node for node, value in qdt.nodes]
        for i, patch in enumerate(patch_sequence):
            # Resize patch to the fixed size expected by the ViT
            resized_patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            final_patches[i] = resized_patch.astype(np.float32)
            
            # Get metadata
            bbox = leaf_nodes[i]
            size, _ = bbox.get_size()
            seq_size[i] = size
            
            x_center = (bbox.x1 + bbox.x2) / 2
            y_center = (bbox.y1 + bbox.y2) / 2
            seq_pos[i] = [x_center, y_center]
            seq_coords[i] = bbox.get_coord()
            
        # 7. Convert everything to PyTorch Tensors
        # Normalize patches from [0, 255] to [0, 1] and convert from HWC to CHW
        patches_tensor = torch.from_numpy(final_patches).permute(0, 3, 1, 2) / 255.0
        
        # Apply standard normalization
        normalize_output = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        patches_tensor = normalize_output(patches_tensor)

        return {
            "patches": patches_tensor,
            "sizes": torch.from_numpy(seq_size),
            "positions": torch.from_numpy(seq_pos),
            "coords": torch.from_numpy(seq_coords),
            "mask": torch.from_numpy(noised_mask),
            "target": target_label,
            "original_image": img_tensor # Return for visualization/debugging if needed
        }
        
# --- Dataloader Builder Function for HDE ---

def build_hde_imagenet_dataloaders(img_size, data_dir, batch_size, fixed_length=512, patch_size=16, num_workers=32):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Standard data augmentation for the base images
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

    # 1. Create the base ImageNet datasets
    print("Loading base ImageNet datasets...")
    # 注意: 代码现在默认使用 ImageNet 数据集。
    base_image_datasets = {x: datasets.ImageNet(root=data_dir, split=x, transform=transform_train if x == 'train' else transform_val) for x in ['train', 'val']}
    print("Base datasets loaded.")
    
    # 2. Wrap them with our custom HDEDataset
    print("Wrapping with HDEDataset...")
    hde_datasets = {x: HDEDataset(base_image_datasets[x], fixed_length=fixed_length, patch_size=patch_size) for x in ['train', 'val']}
    print("HDEDatasets created.")

    # 3. Create samplers and dataloaders
    # Use DistributedSampler if in a distributed environment, otherwise use standard samplers.
    if dist.is_available() and dist.is_initialized():
        print("Using DistributedSampler.")
        samplers = {x: DistributedSampler(hde_datasets[x], shuffle=True) for x in ['train', 'val']}
    else:
        print("Using standard RandomSampler/SequentialSampler.")
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

# --- Example of how to use the dataloader for a full iteration ---
if __name__ == '__main__':
    # This block will only run when the script is executed directly.
    # It demonstrates how to iterate through the entire dataset.
    import time
    parser = argparse.ArgumentParser(description='PyTorch HDE ImageNet DataLoader Example')
    parser.add_argument('--data_dir', type=str, default='/path/to/your/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=1, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--fixed_length', type=int, default=196, help='Number of patches (sequence length)')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of each patch after resizing')
    parser.add_argument('--visualize', action='store_true', help='Generate and save a visualization of one batch')

    args = parser.parse_args()
    
    print("Building HDE ImageNet dataloaders...")
    dataloaders = build_hde_imagenet_dataloaders(
        img_size=args.img_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        fixed_length=args.fixed_length,
        patch_size=args.patch_size,
        num_workers=args.num_workers
    )
    
    print("Dataloaders built.")

    # Optional: Generate visualization before starting the main loop
    if args.visualize:
        print("\nGenerating visualization for one validation batch...")
        try:
            vis_batch = next(iter(dataloaders['val']))
            visualize_batch(vis_batch, save_path="hde_visualization.png")
        except Exception as e:
            print(f"Could not generate visualization: {e}")

    print("Starting iteration...")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        for phase in ['train', 'val']:
            print(f"\nIterating over {phase} set...")
            for step, batch in enumerate(dataloaders[phase]):
                # Our dataloader yields a dictionary
                inputs = batch['patches']
                labels = batch['target']
                
                if step % 100 == 0:
                    # To verify, you can print out the types and shapes
                    print(f"Phase: {phase}, Step: {step}, Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            
            print(f"Finished iterating over {phase} set.")

    total_time = time.time() - start_time
    print(f"\nTotal time cost for loading {args.num_epochs} epoch(s): {total_time:.2f} seconds")
