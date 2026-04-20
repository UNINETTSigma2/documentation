#!/bin/bash -l
#SBATCH --account=nnXXXXk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=8       # change this when more cores are needed
#SBATCH --mem-per-cpu=2GB
#SBATCH --job-name=example
#SBATCH --output=slurm.%j.log

# make the program and environment visible to this script
module restore
module load NRIS/CPU
module load Gaussian/16.C.01-AVX2
module list

# name of input file without extension
input=water

# set the heap-size for the job to 20GB
export GAUSS_LFLAGS2="--LindaOptions -s 20000000"
export PGI_FASTMATH_CPU=avx2

# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/projects/nnXXXXk/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# copy input file to temporary folder and cd into the temporary folder
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR
cd $GAUSS_SCRDIR

# add all nodes to list of known hosts
for HN in $(scontrol show hostnames)
do
    mkdir -p ~/.ssh
    touch ~/.ssh/known_hosts
    chmod 600 ~/.ssh/known_hosts
    ssh-keygen -R "$HN" 2>/dev/null || true
    ssh-keyscan -H "$HN" >> ~/.ssh/known_hosts 2>/dev/null || true
done

# Add line specifying the amount of Linda Workers per node to the input file.
NodeList=$(scontrol show hostnames | while read n ; do echo $n:2 ; done |  paste -d, -s)
{
printf '%%LINDAWORKERS=%s\n' "$NodeList"
cat $input.com
} > $input.tmp && mv $input.tmp $input.com

# run the program
time g16 < $input.com > $input.out

# copy result files back to submit directory
cp $input.out $input.chk $SLURM_SUBMIT_DIR

exit 0
