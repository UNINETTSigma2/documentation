#!/usr/bin/bash

# Assumed to be 'mnist_test.sh'

#SBATCH --account=<your_account>
#SBATCH --job-name=<creative_job_name>
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
## The following line can be omitted to run on CPU alone
#SBATCH --partition=accel --gpus=1
#SBATCH --time=00:30:00

# Purge modules and load tensorflow
module reset
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
# List loaded modules for reproducibility
module list

# Run python script
python $SLURM_SUBMIT_DIR/mnist.py
