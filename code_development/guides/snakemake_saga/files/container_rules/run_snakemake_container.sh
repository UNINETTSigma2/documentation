#!/bin/bash

# Exit immediately if a command fails
set -euo pipefail

# 1. Load the Snakemake module
module reset
module load snakemake/8.27.0-foss-2024a

# Set this to your Saga project account
SBATCH_ACCOUNT=nn????k

# 2. Redirect all caches to the work directory to prevent home quota crashes
export XDG_CACHE_HOME="/cluster/work/users/$USER/.cache"
export APPTAINER_CACHEDIR="/cluster/work/users/$USER/.apptainer_cache"
export SINGULARITY_CACHEDIR="/cluster/work/users/$USER/.apptainer_cache"
mkdir -p "$XDG_CACHE_HOME" "$APPTAINER_CACHEDIR" logs/slurm

echo "Starting Snakemake orchestrator on the login node..."

# 3. Unlock the directory in case of previous crashes
snakemake --unlock --cores 1

# 4. Execute the workflow using containers and Slurm generic cluster submission
snakemake \
    --jobs 50 \
    --use-apptainer \
    --apptainer-args "--cleanenv -B /cluster" \
    --executor cluster-generic \
    --cluster-generic-submit-cmd "sbatch --account=$SBATCH_ACCOUNT \
    --job-name={rule} --parsable --output=logs/slurm/%x-%j.out \
    --error=logs/slurm/%x-%j.err --time={resources.runtime} \
    --cpus-per-task={threads} --mem={resources.mem_mb}M" \
    --default-resources mem_mb=4096 threads=1 runtime=60 \
    --printshellcmds
