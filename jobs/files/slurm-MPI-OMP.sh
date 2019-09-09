#!/bin/bash

#######################################
# example for a hybrid MPI OpenMP job #
#######################################

## Project: replace XXXX with your account ID
#SBATCH --account=nnXXXXk 

## Job name:
#SBATCH --job-name=MyHybridJob
## Allocating amount of resources:
#SBATCH --nodes=10
## Number of tasks (aka processes) to start on each node: 
#SBATCH --ntasks-per-node=2
## Set OMP_NUM_THREADS
#SBATCH --cpus-per-task=16
## Run for a day, syntax is d-hh:mm:ss
#SBATCH --time=1-0:0:0 

# turn on all mail notification, and also provide mail address:
#SBATCH --mail-type=ALL
#SBATCH -M my.email@institution.no

# you may not place bash commands before the last SBATCH directive
######################################################
## Setting variables and prepare runtime environment:
##----------------------------------------------------
## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

# This may or may not be necessary, should be set by SLURM:
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Loading Software modules
# Allways be explicit on loading modules and setting run time environment!!!
module restore system   # Restore loaded modules to the default
module load MySoftWare/Versions #nb: Versions is important!

# Type "module avail MySoftware" to find available modules and versions
# It is also recommended to to list loaded modules, for easier debugging:
module list  

#######################################################
## Prepare jobs, moving input files and making sure 
# output is copied back and taken care of
##-----------------------------------------------------

# Prepare input files
cp inputfiles $SCRATCH
cd $SCRATCH

# Make sure output is copied back after job finishes
savefile outputfile1 outputfile2

########################################################
# Run the application, and we typically time it:
##------------------------------------------------------
 
# Run the application  - please add hash in front of srun and remove 
# hash in front of mpirun if using intel-toolchain 

# For OpenMPI (foss and iomkl toolchains), srun is recommended:
time srun MySoftWare-exec

## For IntelMPI (intel toolchain), mpirun is recommended:
#time mpirun MySoftWare-exec

#########################################################
# That was about all this time; lets call it a day...
##-------------------------------------------------------
# Finish the script
exit 0
