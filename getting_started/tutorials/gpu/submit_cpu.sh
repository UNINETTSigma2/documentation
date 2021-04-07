#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=512M
#SBATCH --qos=devel

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
module list

python gpu_intro.py
