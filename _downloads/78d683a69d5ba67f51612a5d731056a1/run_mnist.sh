#!/usr/bin/bash

#SBATCH --account=<project account>
#SBATCH --job-name=<fancy_name>
#SBATCH --partition=accel --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00

# Purge modules and load tensorflow
module purge
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
# List loaded modules for reproducibility
module list

# Run python script
python $SLURM_SUBMIT_DIR/mnist.py
