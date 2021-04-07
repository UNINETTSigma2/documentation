#!/bin/bash
#SBATCH --job-name=TestGPUOnSaga
#SBATCH --account=nn<XXXX>k
#SBATCH --time=05:00
#SBATCH --mem-per-cpu=512M
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gres=gpu:1

## Set up job environment:
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
module list

# Setup monitoring
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory \
	--format=csv --loop=1 > "gpu_util-$SLURM_JOB_ID.csv" &
NVIDIA_MONITOR_PID=$!  # Capture PID of monitoring process
# Run our computation
python gpu_intro.py
# After computation stop monitoring
kill -SIGINT "$NVIDIA_MONITOR_PID"
