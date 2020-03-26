#!/bin/bash
#SBATCH --account=YourProject
#SBATCH --time=1:0:0
#SBATCH --mem-per-cpu=4G --ntasks=2
#SBATCH --array=1-200

set -o errexit # exit on errors
set -o nounset # treat unset variables as errors
module --quiet purge   # clear any inherited modules

DATASET=dataset.$SLURM_ARRAY_TASK_ID
OUTFILE=result.$SLURM_ARRAY_TASK_ID

YourProgram $DATASET > $OUTFILE
