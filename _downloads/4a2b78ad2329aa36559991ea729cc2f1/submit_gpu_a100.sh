#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=devel
#SBATCH --partition=a100
#SBATCH --gpus=1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --force swap StdEnv Zen2Env
module --quiet purge  # Reset the modules to the system default
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

python gpu_intro.py