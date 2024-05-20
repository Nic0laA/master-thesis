#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=peregrine_attention_unet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# Activate your environment
# source activate peregrine
python="/home/scur2012/Thesis/Peregrine/.venv/bin/python3.12"

# Check whether the GPU is available
srun ${python} -uc "import torch; print('GPU available?', torch.cuda.is_available())"

# Current directory
pwd=`pwd`
current_folder=`basename -a ${pwd}`
save_directory=/scratch-shared/scur2012/final_peregrine_runs/${current_folder}

# Configuration file
config_file=/gpfs/home3/scur2012/Thesis/master-thesis/final_peregrine_runs/config_lowSNR_bhardwaj2023.txt

# Copy config file (change directories)
sed "s/run_id .*$/run_id = ${current_folder}/" ${config_file} | \
sed "s|store_path .*$|store_path = ${save_directory}|" | \
sed "s|obs_path .*$|obs_path = ${save_directory}\/observation_${current_folder}|" > config.txt

mkdir -p ${save_directory}

cp -f ${pwd}/inference_utils.py ${save_directory}/inference_utils.py
cp -f ${pwd}/config.txt ${save_directory}/config.txt

# Move to scratch directory
cd $TMPDIR
if [ ! -d peregrine ]; then
    mkdir peregrine
else
    rm -rf peregrine/*
fi
cd peregrine

# Copy python scripts
cp -f /gpfs/home3/scur2012/Thesis/master-thesis/final_peregrine_runs/peregrine_scripts/{config_utils.py,coverage.py,generate_observation.py,load_simulator.py,run_parallel.py,sampling.py,simulator_utils.py,tmnre.py} .
cp -f ${pwd}/inference_utils.py .
cp -f ${pwd}/config.txt .

srun ${python} generate_observation.py config.txt

srun ${python} tmnre.py config.txt

srun ${python} coverage.py config.txt 20000

