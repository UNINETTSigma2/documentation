#!/bin/bash
#SBATCH --account=nnXXXXk         # Your project
#SBATCH --job-name=GPU_testjob    # A name for your GPU job
#SBATCH --time=0-0:05:00          # Total maximum runtime (e.g., 5 minutes)
#SBATCH --nodes=1                 # Number of nodes to use
#SBATCH --ntasks-per-node=128     # Number of tasks per node
#SBATCH --cpus-per-gpu=70         # 70 CPU cores per GPU (all cores of one Grace-Hopper card)
#SBATCH --mem-per-gpu=110G        # Amount of CPU memory per GPU allocated
#SBATCH --partition=accel         # Run on the GPU accelerator partition
#SBATCH --gpus=1                  # Number of GPUs for the whole job (alternatively use --gpus-per-node=1)
#SBATCH --output=my-job-name-%j.out  # Sets name of output file (%j is replaced by job ID)
#SBATCH --error=my-job-name-%j.err   # Sets name of error file (%j is replaced by job ID)

# Prevent silent errors
set -errext -nounset -o pipefail

# Load necessary modules for CUDA
module load NRIS/GPU
module load CUDA

# Run your GPU application
srun ./my_gpu_program
