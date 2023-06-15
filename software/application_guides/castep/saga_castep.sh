#!/bin/bash -l

#SBATCH --account=nnNNNNk
#SBATCH --job-name=Ethene
#SBATCH --time=0-00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=devel

## Recommended safety settings:
set -o errexit # Make bash exit on any error
set -o nounset # Treat unset variables as errors

# make the program and environment visible to this script
module --quiet purge
module load CASTEP/22.1.1-intel-2022b

# Naming the project:
input=ethene

# create the temporary folder
export tempdir=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $tempdir

# copy input file to temporary folder
cd $SLURM_SUBMIT_DIR
cp *.cell *.param $tempdir
# If you need pseudopotentials, add this line:
# cp *.usp $tempdir

# run the program
cd $tempdir
time srun -n 40 castep.mpi $input

# copy result files back to submit directory
cp $input.castep $SLURM_SUBMIT_DIR

# Also rememeber to copy other files if you want to keep them.
# For instance the $input.check file, the $input.bands file etc.
# The $temp folder is eligible for deletion after a certain amount
# since it resides in $USERWORK.

exit 0
