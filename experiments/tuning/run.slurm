#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=tune_transformer_mvts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%A.out

# Select python version
python="/home/scur2012/.conda/envs/peregrine/bin/python3.11"

# Check whether the GPU is available
srun ${python} -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Run python
run="transformer_mvts"
cd /home/scur2012/Thesis/master-thesis/experiments/tuning
/home/scur2012/.conda/envs/peregrine/bin/jupyter nbconvert --to script ${run}.ipynb
srun ${python} -u ${run}.py
