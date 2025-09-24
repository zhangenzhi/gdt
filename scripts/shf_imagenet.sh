#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o imagenet_shf-vit-b.o%J
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH -p batch
#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

# --- Environment Setup for ROCm/MIOPEN on your HPC ---
export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# --- Load Required Modules ---
# This part is specific to your HPC environment (e.g., Frontier, Crusher)
echo "Loading modules..."
module load PrgEnv-gnu
module load gcc-native/12.3
module load rocm/6.2.0
echo "Modules loaded."

# --- Distributed Training Launch Command ---
# This command uses `srun` to launch the training script in a distributed manner.
# -N 8: Use 8 nodes.
# -n 64: Run a total of 64 processes (ranks).
# --ntasks-per-node 8: Run 8 processes on each node (typically one per GPU).
#
# The main.py script will automatically detect the SLURM environment variables
# set by srun (like SLURM_PROCID) and initialize DDP.
#
# Note: Hyperparameters like epochs and batch size are now controlled by the config file.
echo "Launching distributed training..."
srun -N 2 -n 16 --ntasks-per-node 8 python ./main.py \
    --config ./configs/shf-vit-b_IN1K.yaml \
    --data_dir /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
    --savefile shf-vit-b-n4-bz4k-wk0 \
    --num_workers 0 \
    --reload

# srun -N 2 -n 16 --ntasks-per-node 8 python ./main.py \
#     --config ./configs/vit-b_IN1K.yaml \
#     --data_dir /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
#     --savefile vit-b-n2-bz4k-he-timm-aa \
#     --num_workers 32 \
#     --reload
