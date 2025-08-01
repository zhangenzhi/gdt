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

## Referrence

_A summary of relevant long sequence training methods that reduce the amount of work. *N* = sequence length._

| **Approach** | **Method** | **Merits & Demerits** | **Complexity (Best)** | **Model** | **Implementation** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Attention Approximation | Longformer [beltagy2020longformer] ETC [ainslie2020etc] | **(+)** Better time complexity vs Transformer. \ **(-)** Sparsity levels insufficient for gains to materialize. | O(N) \ O(N√N) | Some Models w/ Forked PyTorch | Custom Self-attention Implementation |
| | BigBird [zaheer2020big] Reformer [kitaev2020reformer] | **(+)** Theoretically proven time complexity. \ **(-)** High-order derivatives | O(NlogN) | Some Models w/ Forked PyTorch | Custom Self-attention Implementation |
| | Sparse Attention [child2019generating] | **(+)** Introduced sparse factorizations of the attention. \ **(-)** Higher time complexity. | O(N√N) | Some Models w/ Forked PyTorch | Custom Self-attention Implementation |
| | Linformer [katharopoulos2020transformers] Performer [choromanski2020rethinking] | **(+)** Fast adaptation \ **(-)** Assumption that self-attention is low rank. | O(N) | Some Models w/ Forked PyTorch | Custom Self-attention Implementation |
| | SPFormer [mei2024spformer] (Prediction) | **(+)** Irregular tokens. \ **(-)** No adaptation to high resolution. | O(P²) \ P:num of regions | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| Hierarchical | Hier. Transformer [si21] (Text Classification) | **(+)** Independent hyperpara. tuning of hierarc. models. \ **(-)** No support for ViT. | O(NlogN) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| | CrossViT [chen2021crossvit] (Classification) | **(+)** Better time complexity vs standard ViT.  \ **(-)** Complex token fusion scheme in dual-branch ViTs. | O(N) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| | HIPT [Chen22] (Classification) | **(+)** Model inductive biases of features in the hierarchy.  \ **(-)** High cost for training multiple models. | O(NlogN) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| | MEGABYTE [yu2023megabyte] (Prediction) | **(+)** Support of multi-modality. \ **(-)** High cost for training multiple models. | O(N<sup>4/3</sup>) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| High-resolution | xT [gupta2024xt] (Prediction) | **(+)** Support of 8K resolution . \ **(-)** Lack of adaptivation. | O(NlogN) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| | HiFormer [heidari2023hiformer] (Prediction) | **(+)** Support of multi-modality. \ **(-)** High cost for ultra-resolution. | O(NlogN) | Custom Model w/ Plain PyTorch | Custom Model Implementation |
| **Ours** | **Gumble Differentialbe Tree** (Segmentation & Class.) | **(+)** Attention mechanism intact.  \ **(+)** Largely reduces computation cost; maintains quality.  \ **(+)** Efficiency depends on level of details in an image. \ **(-)** task semantics are independent of edge information. | O(log²N) | Any Model w/ Plain PyTorch | Image Pre-processing |
