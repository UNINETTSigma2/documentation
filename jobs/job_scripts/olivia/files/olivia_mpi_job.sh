#!/bin/bash
#SBATCH --account=nnXXXXk         # Your project
#SBATCH --job-name=Mandelbrot_MPI # A name for your MPI job
#SBATCH --time=0-0:05:00          # Total maximum runtime (e.g., 5 minutes)
#SBATCH --nodes=1                 # Number of nodes to use
#SBATCH --ntasks-per-node=128     # Number of tasks per node
#SBATCH --cpus-per-task=1         # 1 CPU core per MPI task (adjust if using hybrid MPI/OpenMP)
#SBATCH --mem-per-cpu=3G          # Amount of CPU memory per CPU core
#SBATCH --partition=normal        # Run on the normal CPU partition
#SBATCH --output=my-job-name-%j.out  # Sets name of output file (%j is replaced by job ID)
#SBATCH --error=my-job-name-%j.err   # Sets name of error file (%j is replaced by job ID)

# Prevent silent errors
set -errext -nounset -o pipefail

# Load necessary modules for OpenMPI. Change to NRIS/GPU if on the accel partition
module load NRIS/CPU
module load OpenMPI

# Run your MPI application
srun ./my_mpi_program
