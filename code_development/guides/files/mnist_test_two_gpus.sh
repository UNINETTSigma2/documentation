#!/usr/bin/bash

#SBATCH --account=<your_account>     
#SBATCH --job-name=<your_job_name>
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=accel 
#SBATCH --nodes=1            
#SBATCH --ntasks=1           
#SBATCH --gpus=2             
#SBATCH --cpus-per-task=12   
#SBATCH --mem-per-cpu=8G     
#SBATCH --time=00:30:00

# Reset modules and load tensorflow
module reset
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# list modules for reproducibility
module list

# Run python script
srun python $SLURM_SUBMIT_DIR/mnist_multiple_gpus.py
