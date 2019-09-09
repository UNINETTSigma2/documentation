#!/bin/bash

## Job name:
#SBATCH --job-name=MyLargeJob
## Allocating amount of resources:
#SBATCH --nodes=4
## Number of tasks (aka processes) to start on each node: Pure mpi, one task per core
#SBATCH --ntasks-per-node=32
## No memory pr task since this option is turned off on Fram in QOS normal.
## Run for 10 minutes, syntax is d-hh:mm:ss
#SBATCH --time=0-00:10:00 


cd ${SLURM_SUBMIT_DIR}

# first set of parallel runs
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &

wait

# here a post-processing step
# ...

# another set of parallel runs
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &
mpirun -n 16 ./my-binary &

wait

exit 0
