#!/bin/bash

#####################
# job-array example #
#####################

## Substitute with your project name:
#SBATCH --account=YourProject

#SBATCH --job-name=array_example

# 16 jobs will run in this array at the same time
#SBATCH --array=1-16

# each job will run for maximum five minutes
#              d-hh:mm:ss
#SBATCH --time=0-00:05:00

# you must not place bash commands before the last #SBATCH directive

## Set safer defaults for bash
set -o errexit
set -o nounset

module --quiet purge  # Clear any inherited modules

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python test.py > output_${SLURM_ARRAY_TASK_ID}.txt
