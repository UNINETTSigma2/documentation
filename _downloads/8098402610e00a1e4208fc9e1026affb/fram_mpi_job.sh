#!/bin/bash

###############################################
# Script example for a normal MPI job on Fram #
###############################################

## Project: replace XXXX with your project ID
#SBATCH --account=nnXXXXk 

## Job name:
#SBATCH --job-name=MyJob
## Allocating amount of resources:
#SBATCH --nodes=10
## Number of tasks (aka processes) to start on each node: Pure mpi, one task per core
#SBATCH --ntasks-per-node=32
## No memory pr task since this option is turned off on Fram in partition normal.
## Run for 10 minutes, syntax is d-hh:mm:ss
#SBATCH --time=0-00:10:00 

# you may not place bash commands before the last SBATCH directive
######################################################
## Setting variables and prepare runtime environment:
##----------------------------------------------------
## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

# Loading Software modules
# Allways be explicit on loading modules and setting run time environment!!!
module --quiet purge            # Restore loaded modules to the default
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
