#!/bin/bash

#SBATCH --account=YourProject  # Substitute with your project name
#SBATCH --job-name=parallel_tasks_node
#SBATCH --nodes=4
#SBATCH --time=00:05:00

# Safety settings
set -o errexit
set -o nounset

# Load MPI module
module purge
module load OpenMPI/3.1.1-GCC-7.3.0-2.30
module list

# This is needed for job types that hand out whole nodes:
export SLURM_MEM_PER_CPU=1920

# The set of parallel runs:
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &
srun --ntasks=16 --exclusive ./my-binary &

wait
