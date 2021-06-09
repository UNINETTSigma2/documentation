#!/bin/bash -l
#SBATCH --account=nnXXXXk
#SBATCH --job-name=example
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=32G
#SBATCH --output=slurm.%j.log

# make the program and environment visible to this script
module load Gaussian/g16_C.01

export PGI_FASTMATH_CPU=skylake

# name of input file without extention
input=water

# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# copy input file to temporary folder
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR

# run the program
cd $GAUSS_SCRDIR
time g16.ib $input.com > $input.out

# copy result files back to submit directory
cp $input.out $input.chk $SLURM_SUBMIT_DIR

exit 0
