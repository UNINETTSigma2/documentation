#!/bin/bash

#SBATCH --account=<your-account>
#SBATCH --job-name=example
#SBATCH --partition=normal
#SBATCH --mem=1G
#SBATCH --ntasks=1
#SBATCH --time=00:02:00

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

module reset
module load foss/2022a

# compile and run inside the Slurm job
gcc -O2 simple.c -o simple_c
./simple_c > simple_c.out
