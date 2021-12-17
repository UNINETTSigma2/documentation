#!/bin/bash

#SBATCH --account=YourProject  # Substitute with your project name
#SBATCH --job-name=parallel_tasks_cpu
#SBATCH --ntasks=20
#SBATCH --time=0-00:05:00
#SBATCH --mem-per-cpu=2000M

# Safety settings
set -o errexit
set -o nounset

# Load MPI module
module --quiet purge
module load OpenMPI/4.1.1-GCC-11.2.0
module list

# This is needed with the current version of Slurm (20.11.x):
export SLURM_JOB_NUM_NODES=1-$SLURM_JOB_NUM_NODES

# The set of parallel runs:
srun --ntasks=4 --exact ./my-binary &
srun --ntasks=4 --exact ./my-binary &
srun --ntasks=4 --exact ./my-binary &
srun --ntasks=4 --exact ./my-binary &
srun --ntasks=4 --exact ./my-binary &

wait
