#!/usr/bin/bash

#SBATCH --account=<your_account>
#SBATCH --job-name=<your_job_name>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=a100 --gpus=1
#SBATCH --time=00:30:00

module reset

module load Zen2Env

module load EESSI/2025.06

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# list modules for reproducibility
module list

# Run python script
python $SLURM_SUBMIT_DIR/<your_python_script>.py
