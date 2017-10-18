# Sample Batch Script

Here is a sample batch script that demonstrates usage of various variables and processes for a **normal** job.

```
#!/bin/bash
## Project:
#SBATCH --account=nnNNNNk
## Job name:
#SBATCH --job-name=myjob
## Wall time limit:
#SBATCH --time=1-0:0:0
## Number of nodes:
#SBATCH --nodes=10
## Number of tasks to start on each node:
#SBATCH --ntasks-per-node=2
## Set OMP_NUM_THREADS
#SBATCH --cpus-per-task=16

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

## Software modules
module restore system   # Restore loaded modules to the default
module load mysoftware
module list             # List loaded modules, for easier debugging

## Prepare input files
cp inputfiles $SCRATCH
cd $SCRATCH

## Make sure output is copied back after job finishes
copyback outputfile1 outputfile2

## Run the application
srun mysoftware
```

The actual startup of MPI application differs for different MPI libraries. Since
this part is crucial for application performance, please read about [how to start MPI jobs on Fram](mpi_jobs.md).