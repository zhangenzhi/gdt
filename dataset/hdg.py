import os
import random
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_mean_std(image_paths, img_size, is_main_process):
    """
    遍历所有训练图像，计算单通道的均值和标准差。
    
    Args:
        image_paths (list): 训练图像的路径列表。
        img_size (int): 图像的目标尺寸。
        is_main_process (bool): 是否为主进程 (rank 0)。
        
    Returns:
        tuple: (mean, std)
    """
    if is_main_process:
        print("正在计算训练数据集的均值和标准差...")
    
    total_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0

    # 仅在主进程显示tqdm进度条
    iterator = tqdm(image_paths, desc="正在计算统计数据", disable=not is_main_process)
    for path in iterator:
        img = Image.open(path).convert('L')
        img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        img_np = np.array(img, dtype=np.float64) / 255.0

        sum_pixels += np.sum(img_np)
        sum_sq_pixels += np.sum(np.square(img_np))
        total_pixels += img_np.size

    mean = sum_pixels / total_pixels
    std = np.sqrt(sum_sq_pixels / total_pixels - np.square(mean))

    if is_main_process:
        print("计算完成。")
        print(f"计算出的均值 (Mean): {mean:.4f}")
        print(f"计算出的标准差 (Std): {std:.4f}")
        
    return mean, std

class HydrogelDataset(Dataset):
    """
    用于水凝胶图像分割的数据集类。
    """
    def __init__(self, image_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.mean = 0.0
        self.std = 1.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert("L"))

        mask_path = img_path.replace('revised-hydrogel', os.path.basename(self.mask_dir))
        
        # --- **关键修正** ---
        # 1. 加载原始的uint8蒙版图像
        mask_raw = np.array(Image.open(mask_path).convert("L"))
        # 2. 正确地进行二值化：任何非0像素都为1，0像素保持为0
        # 3. 确保最终类型是float32
        mask = (mask_raw > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)

        return image, mask

def get_transforms(img_size, mean, std, is_train=True):
    """获取训练或验证所需的数据增强流程。"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            # A.OneOf([
            #     A.ElasticTransform(p=0.3),
            #     A.GridDistortion(p=0.3),
            #     A.OpticalDistortion(p=0.3)
            # ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
            ], p=1.0),
            A.Normalize(mean=(mean,), std=(std,)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(mean,), std=(std,)),
            ToTensorV2(),
        ])

def create_dataloaders(data_dir, img_size, batch_size, num_workers=4, use_ddp=False):
    """
    创建支持DDP的训练和验证数据加载器。
    """
    image_dir = os.path.join(data_dir, 'revised-hydrogel')
    mask_dir = os.path.join(data_dir, 'masks-hydrogel')
    all_image_paths = sorted(glob(os.path.join(image_dir, '*/*.jpg')))
    
    # val_specific_img = os.path.join(image_dir, '1301100nm', '19.jpg')
    val_specific_img = os.path.join(image_dir, '1301100nm', '1.jpg')
    if val_specific_img not in all_image_paths:
        raise FileNotFoundError(f"指定的验证图片未找到: {val_specific_img}")

    val_paths = [val_specific_img]
    remaining_paths = [p for p in all_image_paths if p != val_specific_img]
    random.seed(42)
    val_paths.extend(random.sample(remaining_paths, 6))
    train_paths = [p for p in all_image_paths if p not in val_paths]

    is_main_process = not use_ddp or (use_ddp and dist.get_rank() == 0)
    
    if is_main_process:
        print(f"数据集总数: {len(all_image_paths)}")
        print(f"训练集数量: {len(train_paths)}")
        print(f"验证集数量: {len(val_paths)}")
    
    mean, std = calculate_mean_std(train_paths, img_size, is_main_process)

    train_dataset = HydrogelDataset(
        image_paths=train_paths,
        mask_dir=mask_dir,
        transform=get_transforms(img_size, mean, std, is_train=True)
    )
    # 附加均值和标准差，供后续使用
    train_dataset.mean = mean
    train_dataset.std = std
    
    val_dataset = HydrogelDataset(
        image_paths=val_paths,
        mask_dir=mask_dir,
        transform=get_transforms(img_size, mean, std, is_train=False)
    )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_paths), # 加载所有验证图像
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

# --- 本地测试 ---
if __name__ == '__main__':
    # 更新为您的数据路径和参数
    DATA_DIR = '/work/c30636/dataset/hydrogel-s'
    IMG_SIZE = 1024
    BATCH_SIZE = 32 # 1024x1024图像较大，可以适当减小batch size以防内存不足

    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据目录 '{DATA_DIR}' 不存在。请修改 DATA_DIR 变量。")
    else:
        train_loader, val_loader = create_dataloaders(DATA_DIR, IMG_SIZE, BATCH_SIZE)

        print("\n--- Dataloader 测试 ---")
        images, masks = next(iter(train_loader))
        # 期望的尺寸: [Batch, Channel=1, Height, Width]
        print(f"训练批次 - Images shape: {images.shape}, type: {images.dtype}")
        print(f"训练批次 - Masks shape: {masks.shape}, type: {masks.dtype}")
        print(f"图像像素值范围: [{images.min():.2f}, {images.max():.2f}]")
        print(f"蒙版像素值范围: [{masks.min()}, {masks.max()}]")

        # --- 新增：可视化代码 ---
        print("\n--- 可视化测试 ---")
        print("正在为第一个批次生成可视化结果...")

        num_to_show = min(BATCH_SIZE, 32) # 最多显示4张图
        fig, axes = plt.subplots(num_to_show, 2, figsize=(10, num_to_show * 5))
        if num_to_show == 1:
            axes = np.array([axes])

        fig.suptitle("Dataloader 本地测试可视化", fontsize=16)
        
        for i in range(num_to_show):
            img_tensor = images[i]
            mask_tensor = masks[i]
            
            # 将Tensor转换为Numpy array并移除通道维度以便显示
            img_np = img_tensor.numpy().squeeze()
            mask_np = mask_tensor.numpy().squeeze()
            
            # 显示图像
            axes[i, 0].imshow(img_np, cmap='gray')
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')
            
            # 显示蒙版
            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图片而不是直接显示，以兼容无GUI的服务器环境
        save_path = "dataloader_test_visualization.png"
        plt.savefig(save_path)
        print(f"可视化结果已保存到: {save_path}")
