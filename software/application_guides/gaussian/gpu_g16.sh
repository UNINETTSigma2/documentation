#!/bin/bash -l
#SBATCH --account=nnXXXXk
#SBATCH --job-name=example
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gpus=1
#SBATCH --partition=accel
#SBATCH --mem=96G
#SBATCH --output=slurm.%j.log

# make the program and environment visible to this script
module --quiet purge
module load Gaussian/g16_C.01

export PGI_FASTMATH_CPU=skylake

# name of input file without extension
input=caffeine

# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# copy input file to temporary folder
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR

# run the program
cd $GAUSS_SCRDIR
time g16.gpu $input.com > $input.out

# copy result files back to submit directory
cp $input.out $input.chk $SLURM_SUBMIT_DIR

exit 0
