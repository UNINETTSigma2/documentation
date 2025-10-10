#!/bin/bash

#SBATCH --account=YourProject  # Substitute with your project name
#SBATCH --job-name=parallel_tasks_node
#SBATCH --nodes=4
#SBATCH --time=00:05:00

# Safety settings
set -o errexit
set -o nounset

# Load MPI module
module --quiet purge
module load OpenMPI/4.1.1-GCC-11.2.0
module list

# This is needed for job types that hand out whole nodes:
unset SLURM_MEM_PER_NODE
export SLURM_MEM_PER_CPU=1888 # This is for Fram.  For betzy, use 1952.

# This is needed with the current version of Slurm (21.08.x):
export SLURM_JOB_NUM_NODES=1-$SLURM_JOB_NUM_NODES

# The set of parallel runs:
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &
srun --ntasks=16 --exact ./my-binary &

wait
