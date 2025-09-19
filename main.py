import os
import sys
import yaml
import argparse

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training functions from both training scripts
from train.gdt_vit import gdt_imagenet_train, gdt_imagenet_train_local
# --- FIXED: Corrected import to match the function names in the provided script ---
from train.vit_imagenet import vit_imagenet_train, vit_imagenet_train_single
from train.shf_imagenet import shf_imagenet_train, shf_imagenet_train_single

from train.mae_s8d import mae_pretrain_s8d

def main():
    parser = argparse.ArgumentParser(description='Launcher for Vision Transformer Training')
    
    # Core arguments that define the experiment
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file for the specified task.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory for logs and models.')
    parser.add_argument('--savefile', type=str, default=None, help='Subdirectory name for this run. If not provided, uses task_name from config.')
    parser.add_argument('--data_dir', type=str, default='/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--local', type=bool, default=False, help='local model')
    
    # Execution flags
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists.')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for DataLoader.')

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    task_name = config.get('task_name', "others")
    # Use the savefile from args if provided, otherwise use the task_name
    if args.savefile is None:
        args.savefile = task_name
        
    args.output = os.path.join(args.output, task_name)
    os.makedirs(os.path.join(args.output, args.savefile), exist_ok=True)

    is_ddp_environment = 'SLURM_PROCID' 

    if task_name == 'gdt_vit':
        if is_ddp_environment:
            print("--- Launching GDT-ViT in DDP Mode. ---")
            gdt_imagenet_train(args, config)
        else:
            print("--- Launching GDT-ViT in Local Mode. ---")
            gdt_imagenet_train_local(args, config)
    elif task_name == 'vit_imagenet':
        if is_ddp_environment:
            print("--- Launching ViT Baseline in DDP Mode. ---")
            vit_imagenet_train(args, config)
        else:
            print("--- Launching ViT Baseline in Local Mode. ---")
            vit_imagenet_train_single(args, config)
    elif task_name == 'shf_imagenet':
        if is_ddp_environment:
            print("--- Launching ViT Baseline in DDP Mode. ---")
            shf_imagenet_train(args, config)
        else:
            print("--- Launching ViT Baseline in Local Mode. ---")
            shf_imagenet_train_single(args, config)
    elif task_name == 'mae_s8d_pretrain':
        if is_ddp_environment:
            print("--- Launching MAE-ViT in DDP Mode. ---")
            mae_pretrain_s8d(args=args,config=config)
        else:
            print("--- Launching MAE-ViT in Local Mode. ---")
            mae_pretrain_s8d(args=args,config=config)
    else:
        raise ValueError(f"Unknown task: {task_name}")

if __name__ == '__main__':
    main()
