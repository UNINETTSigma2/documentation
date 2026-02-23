#!/usr/bin/bash

#SBATCH --account=<your_account> 
#SBATCH --job-name=<creative_job_name>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel --gpus=2
#SBATCH --time=00:30:00

# Reset modules and load tensorflow
module reset
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# List loaded modules for reproducibility
module list

# Run python script
srun python $SLURM_SUBMIT_DIR/mnist_multiple_gpus.py
