#!/bin/bash -l
#SBATCH --account=nnXXXXk
#SBATCH --job-name=example
#SBATCH --time=0-00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1             ## Linda does not use this variable
#SBATCH --output=slurm.%j.log

# make the program and environment visible to this script
module --quiet reset
module load Gaussian/16.C.01-AVX2

# name of input file without extension
input=water

# set the heap-size for the job to 20GB
export GAUSS_LFLAGS2="--LindaOptions -s 20000000"
export PGI_FASTMATH_CPU=avx2
# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# add all nodes to list of known hosts
for HN in $(scontrol show hostnames "$SLURM_JOB_NODELIST")
do
    mkdir -p ~/.ssh
    touch ~/.ssh/known_hosts
    chmod 600 ~/.ssh/known_hosts
    ssh-keygen -R "$HN" 2>/dev/null || true
done

# copy input file to temporary folder and cd into it
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR
cd $GAUSS_SCRDIR

# Add line specifying the amount of Linda Workers per node to a temporary copy of the input file
# This starts one Linda worker per node.
NodeList=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | while read n ; do echo $n:1 ; done |  paste -d, -s)
{
printf '%%LINDAWORKERS=%s\n' "$NodeList"
cat $input.com
} > job_runtime.com 

#running the calculation
time g16 < job_runtime.com > $input.out

# copy result files back to submit directory
cp $input.out $input.chk $SLURM_SUBMIT_DIR

exit 0
