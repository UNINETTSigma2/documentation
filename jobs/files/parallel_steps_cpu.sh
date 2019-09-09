#!/bin/bash

#SBATCH --account=YourProject  # Substitute with your project name
#SBATCH --job-name=parallel_tasks_cpu
#SBATCH --ntasks=20
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=5000M

# Safety settings
set -o errexit
set -o nounset

# Load MPI module
module --quiet purge
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module list

# The set of parallel runs:
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &
srun --ntasks=4 --exclusive ./my-binary &

wait
