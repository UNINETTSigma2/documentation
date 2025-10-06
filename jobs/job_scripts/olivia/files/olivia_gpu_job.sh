#!/bin/bash
#SBATCH --account=nnXXXXk     # Your project
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=05:10:00
#SBATCH --account=nnXXXXk
#SBATCH --job-name=my-job-name
#SBATCH --output=my-job-name-%j.out
#SBATCH --error=my-job-name-%j.err

# Load necessary modules (e.g., CUDA toolkit and programming environment)
module load PrgEnv-nvidia # Example programming environment
module load cuda/12.2 # Example CUDA version

# Run your GPU application
srun ./my_gpu_program
