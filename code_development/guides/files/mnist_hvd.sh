#!/usr/bin/bash

#SBATCH --account=<your_account>
#SBATCH --job-name=<your_job_name>
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00

# Reset modules and load TensorFlow and Horovod
module reset
module load CMake/3.23.1-GCCcore-11.3.0
module load Horovod/0.28.1-foss-2022a-CUDA-11.7.0-TensorFlow-2.11.0

# Export settings expected by Horovod and MPI
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ob1"
export HOROVOD_MPI_THREADS_DISABLE=1

# Run the Python script using srun
srun python $SLURM_SUBMIT_DIR/mnist_hvd.py
