import os
import sys
import yaml
import argparse

# Ensure the project root is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import training functions from both training scripts
from train.gdt_vit_ddp import gdt_imagenet_train, gdt_imagenet_train_local
# --- FIXED: Corrected import to match the function names in the provided script ---
from train.vit_imagenet_ddp import vit_imagenet_train, vit_imagenet_train_local

def main():
    """
    Main entry point for launching training jobs.
    Parses arguments, loads configuration, and calls the appropriate
    training function based on the specified task.
    """
    parser = argparse.ArgumentParser(description='Launcher for Vision Transformer Training')
    
    # Core arguments that define the experiment
    parser.add_argument('--task', type=str, required=True, choices=['gdt_vit', 'vit_baseline'], help='The training task to run.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file for the specified task.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory for logs and models.')
    parser.add_argument('--savefile', type=str, default=None, help='Subdirectory name for this run. If not provided, uses task_name from config.')
    parser.add_argument('--data_dir', type=str, default='/lustre/orion/nro108/world-shared/enzhi/dataset/imagenet', help='Path to the ImageNet dataset directory')
    
    # Execution flags
    parser.add_argument('--reload', action='store_true', help='Resume training from the best checkpoint if it exists.')
    parser.add_argument('--local', action='store_true', help='Force local (single-GPU) training, even in a SLURM environment.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader.')

    args = parser.parse_args()

    # --- Load Config ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    # --- Prepare output directory ---
    task_name = config.get('task_name', args.task)
    # Use the savefile from args if provided, otherwise use the task_name
    if args.savefile is None:
        args.savefile = task_name
        
    args.output = os.path.join(args.output, task_name)
    os.makedirs(os.path.join(args.output, args.savefile), exist_ok=True)
    
    # --- Decide on execution mode and task ---
    is_ddp_environment = 'SLURM_PROCID' in os.environ and not args.local

    if args.task == 'gdt_vit':
        if is_ddp_environment:
            print("--- Launching GDT-ViT in DDP Mode. ---")
            gdt_imagenet_train(args, config)
        else:
            print("--- Launching GDT-ViT in Local Mode. ---")
            gdt_imagenet_train_local(args, config)
    elif args.task == 'vit_baseline':
        if is_ddp_environment:
            print("--- Launching ViT Baseline in DDP Mode. ---")
            # --- FIXED: Corrected function call ---
            vit_imagenet_train(args, config)
        else:
            print("--- Launching ViT Baseline in Local Mode. ---")
            # --- FIXED: Corrected function call ---
            vit_imagenet_train_local(args, config)
    else:
        raise ValueError(f"Unknown task: {args.task}")

if __name__ == '__main__':
    main()
