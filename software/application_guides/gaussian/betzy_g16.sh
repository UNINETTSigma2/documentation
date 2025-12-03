#!/bin/bash -l
#SBATCH --account=nnXXXXk
#SBATCH --job-name=example
#SBATCH --time=0-00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1             ## Linda does not use this variable
#SBATCH --output=slurm.%j.log

# make the program and environment visible to this script
module --quiet purge
module load Gaussian/16.C.01-AVX2

# name of input file without extension
input=water

# set the heap-size for the job to 20GB
export PGI_FASTMATH_CPU=avx2


# create the temporary folder
export GAUSS_SCRDIR=/cluster/work/users/$USER/$SLURM_JOB_ID
mkdir -p $GAUSS_SCRDIR

# add all nodes to list of known hosts
for HN in $(scontrol show hostnames)
do
    mkdir -p ~/.ssh
    touch ~/.ssh/known_hosts
    chmod 600 ~/.ssh/known_hosts
    ssh-keygen -R "$HN" 2>/dev/null || true
    ssh-keyscan -H "$HN" >> ~/.ssh/known_hosts 2>/dev/null || true
    
    ## Test if the nodes are reachable
    ssh -o BatchMode=yes "$HN" true && echo "SSH OK" || echo "SSH failed"
done


# copy input file to temporary folder
cp $SLURM_SUBMIT_DIR/$input.com $GAUSS_SCRDIR

# run the program
cd $GAUSS_SCRDIR

# Add line specifying the amount of Linda Workers per node.
NodeList=$(scontrol show hostnames | while read n ; do echo $n:1 ; done |  paste -d, -s)
{
printf '%%LINDAWORKERS=%s\n' "$NodeList"
cat $input.com
} > $input.tmp && mv $input.tmp $input.com


time g16 < $input.com > $input.out

# copy result files back to submit directory
cp $input.out $input.chk $SLURM_SUBMIT_DIR

exit 0
