#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
module list

python gpu_intro.py
