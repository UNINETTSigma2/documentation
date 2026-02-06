#!/bin/bash

#SBATCH --account=nn<XXXX>k
#SBATCH --job-name=TestGPUOutput
#SBATCH --time=12:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

#SBATCH --partition=accel
#SBATCH --gpus=1

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet reset  # Reset the modules to the system default
module load hipSYCL/0.9.1-gcccuda-2020b
module list     # For easier debugging

# Activate local virtual environment
source env/bin/activate
# Run our application test on compute node with GPU
pytest
