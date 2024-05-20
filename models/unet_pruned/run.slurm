#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train_unet_pruned
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --array 1-7
#SBATCH --output=slurm_output_%A_%a.out

# Exit script on first error
set -e

# Select python version
python="/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12"

# Check whether the GPU is available
srun ${python} -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Run python

ratios=(0 0.05 0.1 0.2 0.3 0.4 0.5)

srun ${python} -u train_model.py --prune_ratio ${ratios[$SLURM_ARRAY_TASK_ID-1]}
