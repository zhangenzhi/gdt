import os
import argparse

from train.gdt_vit_ddp import gdt_imagenet_train


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--logname', type=str, default='train.log', help='logging of task.')
    parser.add_argument('--output', type=str, default='./output', help='output dir')
    parser.add_argument('--savefile', type=str, default='vit-imagenet', help='output dir')
    parser.add_argument('--gpus', type=int, default=8, help='Epochs for iteration')
    parser.add_argument('--nodes', type=int, default=1, help='Epochs for iteration')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--seq_length', type=int, default=196, help='Epochs for iteration')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained weights')
    parser.add_argument('--reload', type=bool, default=True, help='Reuse previous weights')
    parser.add_argument('--2d', type=bool, default=True, help='Use flat the 3d mri image to 2d.')
    
    args = parser.parse_args()
    return args

def main(args):
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    if args.task == "gdt_imagenet_ddp":
        gdt_imagenet_train(args=args)
    else:
        raise "No such task."
    
if __name__ == '__main__':
    args = parse_args()
    main(args=args)