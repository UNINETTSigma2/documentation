#!/bin/bash

set -euo pipefail

# Load the module (adjust version if needed)

module reset
module load snakemake/8.27.0-foss-2024a

# Set this to your Saga project account
SBATCH_ACCOUNT=nn????k

# Redirect Snakemake cache to work directory to avoid filling home quota
export XDG_CACHE_HOME="/cluster/work/users/$USER/.cache"
mkdir -p "$XDG_CACHE_HOME"

echo "Running Snakemake in Cluster Mode..."

# Create log directory for Slurm output files
mkdir -p logs/slurm

# --cluster "sbatch ..." tells Snakemake to submit each rule as a Slurm job
# --jobs 50 limits the max concurrent jobs to 50
# --default-resources sets defaults for rules that don't have them defined in Snakefile
snakemake --unlock --cores 1
snakemake \
    --jobs 50 \
    --executor cluster-generic \
    --cluster-generic-submit-cmd "sbatch --account=$SBATCH_ACCOUNT --job-name={rule} \
    --parsable --output=logs/slurm/%x-%j.out --error=logs/slurm/%x-%j.err \
    --time={resources.runtime} \
    --cpus-per-task={threads} --mem={resources.mem_mb}M" \
    --default-resources mem_mb=4096 threads=1 runtime=60 \
    --printshellcmds
