#!/usr/bin/bash

#SBATCH --account=<account_name>  # Replace with your account name
#SBATCH --job-name=<job_name>  # Replace with a descriptive job name
#SBATCH --partition=accel
#SBATCH --nodes=2                # Request 2 nodes
#SBATCH --ntasks=8               # Total number of tasks (4 tasks per node)
#SBATCH --gpus-per-task=1        # 1 GPU per task
#SBATCH --mem-per-cpu=8G         # Memory per CPU
#SBATCH --time=00:30:00          # Time limit

# Reset modules and load TensorFlow and Horovod
module reset
module load CMake/3.23.1-GCCcore-11.3.0
module load Horovod/0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.11.0
# List loaded modules for reproducibility
module list
# Export settings expected by Horovod and MPI
export OMPI_MCA_btl="^openib"  # Disable InfiniBand if not configured
export OMPI_MCA_pml="ob1"      # Use the "ob1" communication layer
export HOROVOD_MPI_THREADS_DISABLE=1  # Disable MPI threading for Horovod
export OMPI_MCA_btl_base_verbose=100  # Debug MPI communication

# Run the Python script using srun
srun python $SLURM_SUBMIT_DIR/mnist_hvd.py
