#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=train_transformer_mvts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%A.out

# Select python version
python="/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12"

# Check whether the GPU is available
srun ${python} -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Run python

cd /gpfs/home3/scur2012/Thesis/master-thesis/models/transformer_mvts
srun ${python} -u train_model.py
