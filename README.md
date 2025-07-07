# GDT-ViT: Gated Differentiable Transformer for Vision

This repository contains the official implementation of the GDT-ViT, a novel Vision Transformer architecture featuring a gated differentiable transformer for enhanced performance on image classification tasks.

## Description

The GDT-ViT introduces a gating mechanism to the standard Vision Transformer (ViT) architecture. This allows the model to dynamically control the flow of information through the network, leading to improved performance and efficiency. This implementation provides a framework for training and evaluating GDT-ViT models on the ImageNet dataset. The code is structured to support both distributed training on SLURM clusters and local single-GPU execution.

## Usage

The main entry point for this project is `main.py`, which serves as a launcher for training sessions.

### Training

To start a training session, run the following command:

```bash
python main.py --config [path/to/config.yaml] --data_dir [path/to/dataset]
```

### Key Arguments

*   `--config`: Path to the YAML configuration file (e.g., `configs/gdt_vit.yaml`).
*   `--output`: Base directory to save logs and model checkpoints. Defaults to `./output`.
*   `--savefile`: Subdirectory name for the specific run. Defaults to `run_01`.
*   `--data_dir`: Path to the ImageNet dataset.
*   `--reload`: Use this flag to resume training from the latest checkpoint.
*   `--local`: Force training on a single local GPU, even in a SLURM environment.
*   `--num_workers`: Number of data loader workers. Defaults to 8.

## Configuration

The training process is configured through a YAML file. An example configuration can be found in `configs/gdt_vit.yaml`. This file defines the model architecture, hyperparameters, and other training parameters.

## Training Modes

The training script supports two modes:

1.  **Distributed Data Parallel (DDP) Mode**: This is the default mode when running in a SLURM environment. It leverages multiple GPUs for efficient, large-scale training.
2.  **Local Mode**: This mode is for single-GPU training. It can be forced by using the `--local` flag.

The script automatically detects the environment and selects the appropriate mode.

## Output

All training artifacts, including logs and model checkpoints, are saved to the directory specified by the `--output` and `--savefile` arguments. The final output path will be `[output]/[task_name]/[savefile]`, where `task_name` is defined in the configuration file.

