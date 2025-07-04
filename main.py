import os
import sys
import yaml
import argparse

# Ensure the project root is in the python path
# This allows us to run `python main.py` from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# We import the functions from the training script, which now acts as a library
from train.gdt_vit_ddp import gdt_imagenet_train, gdt_imagenet_train_local

def main():
    """
    Main entry point for launching GDT-ViT training.
    This script parses arguments, loads configuration, and decides whether
    to run in a distributed (DDP) mode for SLURM or a local single-GPU mode.
    """
    parser = argparse.ArgumentParser(description='Launcher for GDT-ViT Training')
    
    # Core arguments that define the experiment
    parser.add_argument('--config', type=str, default='./configs/gdt_vit.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--output', type=str, default='./output', help='Base output directory for logs and models.')
    parser.add_argument('--savefile', type=str, default='run_01', help='Subdirectory name for saving logs and models for this specific run.')
    parser.add_argument('--data_dir', type=str, default='/lustre/orion/nro108/world-shared/enzhi/gdt/dataset', help='Path to the ImageNet dataset directory')
    
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
    # The final output path will be ./output/{task_name_from_config}/{savefile}
    task_name = config.get('task_name', 'default_task')
    args.output = os.path.join(args.output, task_name)
    os.makedirs(os.path.join(args.output, args.savefile), exist_ok=True)
    
    # --- Decide on execution mode ---
    # Check for SLURM environment variables to decide whether to run in DDP mode
    is_ddp_environment = 'SLURM_PROCID' in os.environ and not args.local

    if is_ddp_environment:
        print("--- Detected SLURM environment. Running in DDP Mode. ---")
        gdt_imagenet_train(args, config)
    else:
        print("--- Running in Local Mode (Single GPU, No DDP). ---")
        gdt_imagenet_train_local(args, config)

if __name__ == '__main__':
    main()
