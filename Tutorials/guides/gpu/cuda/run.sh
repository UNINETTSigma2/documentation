#!/bin/bash
#SBATCH --job-name=CUDA-test
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=1G
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load CUDA/11.1.1-GCC-10.2.0
module list

# Compile our code
nvcc loop_add_cuda.cu -o loop_add_cuda

# Run our computation
./loop_add_cuda

exit 0
