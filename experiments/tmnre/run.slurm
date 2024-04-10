#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=peregrine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=48:00:00
#SBATCH --output=slurm_output_%A.out

export DISABLE_TQDM=True

# Select python version
python="/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12"

# Check whether the GPU is available
srun ${python} -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Run python
cd /home/scur2012/Thesis/master-thesis/experiments/tmnre
srun ${python} -u run_tmnre.py
