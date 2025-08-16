import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import os
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

from PIL import Image

def get_val_transform(img_size):
    # 根据提供的 eval transform 逻辑来构建
    t = []
    if img_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
        
    size = int(img_size / crop_pct)
    
    t.append(transforms.Resize(size, interpolation=Image.BICUBIC))
    t.append(transforms.CenterCrop(img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(t)

# 建议将 num_workers 作为参数传入，而不是依赖外部的 args
def imagenet_distribute(img_size, data_dir, batch_size, num_workers=32):
    """
    为分布式训练优化的ImageNet DataLoader函数
    """
    # 数据增强部分保持不变
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size,scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=9, magnitude=15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': get_val_transform(img_size)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a subset of ImageNet for testing.")
    # parser.add_argument('--data_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet", help="Path to the original ImageNet directory containing 'train' and 'val' folders.")
    # parser.add_argument('--output_dir', type=str, default="/lustre/orion/nro108/world-shared/enzhi/gdt/dataset", help="Path to save the generated subset.")
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help="Path to the original ImageNet directory containing 'train' and 'val' folders.")
    parser.add_argument('--output_dir', type=str, default="/work/c30636/gdt/dataset", help="Path to save the generated subset.")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes to include in the subset.")
    parser.add_argument('--train_imgs', type=int, default=500, help="Number of training images per class.")
    parser.add_argument('--val_imgs', type=int, default=100, help="Number of validation images per class.")
    
    args = parser.parse_args()

    # --- Create Training Subset ---
    print("--- Starting Training Subset Creation ---")
    original_train_dir = os.path.join(args.data_dir, 'train')
    subset_train_dir = os.path.join(args.output_dir, 'train')
    create_imagenet_subset(original_train_dir, subset_train_dir, args.num_classes, args.train_imgs)

    # --- Create Validation Subset from the same classes ---
    print("\n--- Starting Validation Subset Creation ---")
    # Get the classes selected for the training set to ensure consistency
    selected_train_classes = os.listdir(subset_train_dir)
    original_val_dir = os.path.join(args.data_dir, 'val')
    subset_val_dir = os.path.join(args.output_dir, 'val')
    
    if os.path.exists(subset_val_dir):
        shutil.rmtree(subset_val_dir)
    os.makedirs(subset_val_dir)
    
    for class_name in tqdm(selected_train_classes, desc="Processing validation classes"):
        subset_class_dir = os.path.join(subset_val_dir, class_name)
        os.makedirs(subset_class_dir)
        
        original_class_dir = os.path.join(original_val_dir, class_name)
        if not os.path.isdir(original_class_dir):
            # print(f"Warning: Class '{class_name}' not found in validation set. Skipping.")
            continue
            
        class_images = [f for f in os.listdir(original_class_dir) if os.path.isfile(os.path.join(original_class_dir, f))]
        k_images = min(args.val_imgs, len(class_images))
        if k_images == 0:
            continue
        
        selected_images = random.sample(class_images, k=k_images)
        
        for image_name in selected_images:
            original_image_path = os.path.join(original_class_dir, image_name)
            subset_image_path = os.path.join(subset_class_dir, image_name)
            shutil.copy(original_image_path, subset_image_path)
            
    print("\nValidation subset creation completed.")
    print("Subset creation finished. Find your data at:", args.output_dir)
    
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


