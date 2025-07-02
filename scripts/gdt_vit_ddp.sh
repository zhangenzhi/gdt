#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o gdt_imagenet_vit_ddp.o%J
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH -p batch
#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

module load PrgEnv-gnu
module load gcc-native/12.3
module load rocm/6.2.0

srun -N 2 -n 16 --ntasks-per-node 8 python main.py \
    --task gdt_imagenet_vit_ddp \
    --data_dir /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
    --batch_size 512 \
    --num_workers 32 \
    --num_epochs 100 \
    --savefile gdt-vit-n2-bz512

###SBATCH -q debug