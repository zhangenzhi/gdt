import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    """
    Builds the ImageNet dataset with a specific transformation pipeline.

    Args:
        is_train (bool): If True, returns the training dataset. Otherwise, returns the validation dataset.
        args (Namespace): An object containing various arguments like data_path, input_size, etc.

    Returns:
        torchvision.datasets.ImageFolder: The dataset object.
    """
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    """
    Builds the image transformation pipeline using timm's create_transform.

    Args:
        is_train (bool): If True, builds a training transformation pipeline with data augmentation.
                         Otherwise, builds a validation transformation pipeline.
        args (Namespace): An object containing various arguments like input_size,
                          auto_augment (aa), random erasing (re_prob), etc.

    Returns:
        transforms.Compose: The composed transformation pipeline.
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    
    # 构建训练转换
    if is_train:
        # 使用timm的create_transform，它会自动处理各种数据增强策略
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # 构建验证转换
    # 这部分逻辑与timm的推荐做法一致
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class ImagenetTransformArgs:
    def __init__(self, **kwargs):
        # 默认参数值，可以根据需要修改
        self.input_size = 224
        self.color_jitter = 0.0
        # self.aa = 'rand-m9-mstd0.5-inc1'
        self.aa = 'rand-m9-mstd0.5'
        self.reprob = 0.0
        self.remode = 'pixel'
        self.recount = 1
        
        # 使用传入的关键字参数覆盖默认值
        for key, value in kwargs.items():
            setattr(self, key, value)