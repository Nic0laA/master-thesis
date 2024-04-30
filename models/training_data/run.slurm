#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=generate_samples
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%A.out

# Select python version
python="/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12"

# Run python
cd /home/scur2012/Thesis/master-thesis/training_data
srun ${python} -u generate_training_data.py --datadir  "/scratch-shared/scur2012/training_data/default_limits_2e6" -n 2000000
