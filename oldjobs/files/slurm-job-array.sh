#!/bin/bash

#####################
# job-array example #
#####################

## Substitute with your project name:
#SBATCH --account=nnNNNNk

#SBATCH --job-name=example

# 16 jobs will run in this array at the same time
#SBATCH --array=1-16

# each job will run for maximum five minutes
#              d-hh:mm:ss
#SBATCH --time=0-00:05:00

# For this example we use the preproc qos so we don't waste nodes:
#SBATCH --qos=preproc

# you must not place bash commands before the last #SBATCH directive

## Set safer defaults for bash
set -o errexit
set -o nounset


## Go to the job's $SCRATCH dir:
cd ${SCRATCH}

## Copy files here:
cp ${SLURM_SUBMIT_DIR}/test.py .

## Mark the output file for automatic copying to $SLURM_SUBMIT_DIR afterwards:
savefile output_${SLURM_ARRAY_TASK_ID}.txt

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python test.py > output_${SLURM_ARRAY_TASK_ID}.txt

# happy end
exit 0