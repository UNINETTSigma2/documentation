#!/bin/bash
#SBATCH --account=nnXXXXk     # Your project
#SBATCH --job-name=Mandelbrot_MPI # A name for your MPI job
#SBATCH --time=0-0:05:00      # Total maximum runtime (e.g., 5 minutes)
#SBATCH --ntasks=255          # Total number of MPI tasks (Maximum ranks with srun is 255, 252 with OpenMPI mpirun)
#SBATCH --nodes=4             # Number of nodes to use
#SBATCH --partition=normal    # Run on the normal CPU partition
#SBATCH --cpus-per-task=1     # 1 CPU core per MPI task (adjust if using hybrid MPI/OpenMP)
#SBATCH --mem-per-cpu=3G      # Amount of CPU memory per CPU core


# Load necessary modules for your compilers and MPI library (e.g., PrgEnv-gnu, cray-mpich)
# module load PrgEnv-gnu cray-mpich # Example modules

# Run your MPI application
srun ./my_mpi_program
