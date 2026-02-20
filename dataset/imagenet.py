import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler, Dataset
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

from dataset.transform import ImagenetTransformArgs, build_transform
def imagenet(args):
    # 数据增强部分保持不变
    data_transforms = {
        'train':build_transform(is_train=True, args=ImagenetTransformArgs(input_size=224)),
        'val': build_transform(is_train=False, args=ImagenetTransformArgs(input_size=224))
    }

    # 创建数据集
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

def imagenet_distribute(img_size, data_dir, batch_size, num_workers=32):
    """
    为分布式训练优化的ImageNet DataLoader函数
    """
    # 数据增强部分保持不变
    data_transforms = {
        'train':build_transform(is_train=True, args=ImagenetTransformArgs(input_size=img_size)),
        'val': build_transform(is_train=False, args=ImagenetTransformArgs(input_size=img_size))
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
#     parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for DataLoader')
#     parser.add_argument('--num_workers', type=int, default=64, help='Number of workers for DataLoader')
    
#     args = parser.parse_args()
    
#     dataloaders = imagenet(args)
    
#     import time
#     for i in range(args.num_epochs):
#         print("Current epochs: {} --------".format(i))
#         start_time = time.time()
#         for phase in ['train', 'val']:
#             for step, (inputs, labels) in enumerate(dataloaders[phase]):
#                 if step % 100 == 0:
#                     # 为了验证，可以打印出变量的类型和形状
#                     print(f"Phase: {phase}, Step: {step}, Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")

#         print("Time cost for loading {}".format(time.time() - start_time))

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
    
if __name__ == '__main__':
    import time
    parser = argparse.ArgumentParser(description='SHF Quadtree Dataloader with Timm Augmentation Test')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=48, help='Number of workers for DataLoader.')
    parser.add_argument('--visualize', action='store_true', help='Generate and save a visualization of one batch.')
    args = parser.parse_args()
    
    img_size = 256
    dataloaders = build_shf_imagenet_dataloader(
        img_size=img_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n--- Dataloader Sanity Check ---")
    
    if args.visualize:
        from gdt.shf import denormalize, deserialize_patches
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        print("Fetching one batch from the validation set for visualization...")
        batch_dict, labels = next(iter(dataloaders['val']))
        print("Batch fetched successfully!")
        
        original_img_tensor = batch_dict['original_image'][0]
        patches_tensor = batch_dict['patches'][0]
        coords_tensor = batch_dict['coords'][0]

        original_image_np = denormalize(original_img_tensor)
        reconstructed_image_np = deserialize_patches(patches_tensor, coords_tensor, img_size)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("SHF Dataloader Visualization Check", fontsize=16)

        axes[0].imshow(original_image_np)
        axes[0].set_title("Original Augmented Image")
        axes[0].axis('off')

        axes[1].imshow(original_image_np)
        axes[1].set_title("Quadtree Grid")
        axes[1].axis('off')
        for i in range(coords_tensor.shape[0]):
            x1, x2, y1, y2 = coords_tensor[i].numpy()
            if x2 - x1 > 0:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='cyan', facecolor='none')
                axes[1].add_patch(rect)

        axes[2].imshow(reconstructed_image_np)
        axes[2].set_title("Reconstructed Image")
        axes[2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = "shf_dataloader_visualization.png"
        plt.savefig(save_path)
        print(f"\n✅ Visualization saved to: {save_path}")

    else:
        print("--- Full Iteration Speed Test ---")
        
        for phase in ['train', 'val']:
            print(f"\nIterating over {phase} set...")
            num_batches = len(dataloaders[phase])
            data_iter = iter(dataloaders[phase])
            
            if num_batches < 2:
                print("Not enough batches to perform a warm-up and benchmark. Skipping speed test.")
                continue

            # --- Warm-up Phase ---
            print("  Processing first batch (includes worker startup cost)...")
            start_warmup = time.time()
            batch_dict, labels = next(data_iter)
            end_warmup = time.time()
            
            first_batch_time = end_warmup - start_warmup
            current_batch_size = labels.shape[0]
            time_per_image_warmup = first_batch_time / current_batch_size if current_batch_size > 0 else 0
            print(f"  Warm-up batch 1/{num_batches} processed. | "
                  f"Batch time: {first_batch_time:.8f}s | "
                  f"Time/image: {time_per_image_warmup * 1000:.8f}ms")

            # --- Benchmarking Phase ---
            print("  Starting main benchmark loop...")
            
            total_benchmark_time = 0.0
            total_images_processed = 0
            batch_start_time = time.time() # Initialize for the first benchmark batch

            for i in range(1, num_batches):
                # Fetching the batch is the main work we want to time
                batch_dict, labels = next(data_iter)
                batch_end_time = time.time()

                # Calculate time for this specific batch and accumulate
                batch_time = batch_end_time - batch_start_time
                current_batch_size = labels.shape[0]
                total_benchmark_time += batch_time
                total_images_processed += current_batch_size
                
                if (i + 1) % 100 == 0 or i == num_batches - 1:
                    # This print is just for progress, not for timing
                    print(f"  Processed batch {i + 1}/{num_batches}, current batch time: {batch_time}")

                # Reset timer for the next batch
                batch_start_time = time.time()
            
            # --- Report Results ---
            avg_batch_time = total_benchmark_time / (num_batches - 1) if num_batches > 1 else 0
            avg_time_per_image = total_benchmark_time / total_images_processed if total_images_processed > 0 else 0
            
            print(f"\n  --- {phase.capitalize()} Set Benchmark Results (excluding first batch) ---")
            print(f"  Total benchmark time: {total_benchmark_time:.8f}s for {num_batches - 1} batches")
            print(f"  Total images processed: {total_images_processed}")
            print(f"  Average batch time: {avg_batch_time:.8f}s")
            print(f"  Overall average time per image: {avg_time_per_image * 1000:.8f}ms")

        print(f"\n✅ Full iteration completed.")