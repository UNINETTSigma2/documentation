(pytorch-multi-node)=

# Multi-Node Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 3 of the PyTorch on Olivia guide. See {ref}`pytorch-on-olivia` for single-GPU and {ref}`pytorch-multi-gpu` for multi-GPU setup.

Multi-node training on Olivia requires proper configuration of NCCL with the OFI plugin and libfabric for the Slingshot interconnect. The job script below handles these configurations.



## Job Script for Multi-Node Training

```{code-block} bash
:linenos:
:emphasize-lines: 8, 10, 17, 20-22, 25-29, 43-47, 58-75, 78-84

#!/bin/bash
#SBATCH --account=nn9997k
#SBATCH --job-name=resnet_multinode
#SBATCH --output=multinode_%j.out
#SBATCH --error=multinode_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G

# Path to the container
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"

# Path to the training script
export APPTAINERENV_TRAINING_SCRIPT="train_ddp.py --epochs 100 --batch-size 2048 --base-lr 0.04 --target-accuracy 0.95 --patience 2"

# Load communication libraries from the NRIS GPU software stack
module purge
module load NRIS/GPU
module load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0  # pulls in aws-ofi-nccl + libfabric

# Resolve bind-mount source paths from module environment variables
HOST_LIBFABRIC_LIB_PATH="$EBROOTLIBFABRIC/lib"
HOST_LIBFABRIC_INCLUDE_PATH="$EBROOTLIBFABRIC/include"
HOST_NCCL_LIB_PATH="$EBROOTNCCL/lib"
HOST_AWS_OFI_LIB_PATH="$EBROOTAWSMINOFIMINNCCL/lib"
HOST_CXI_LIB_PATH=/usr/lib64  # libcxi.so.1 lives here; unchanged

# Validate host paths before launching the container
for p in "$HOST_LIBFABRIC_LIB_PATH" "$HOST_LIBFABRIC_INCLUDE_PATH" \
      "$HOST_NCCL_LIB_PATH" "$HOST_AWS_OFI_LIB_PATH" "$HOST_CXI_LIB_PATH"; do
  if [ ! -d "$p" ]; then
    echo "ERROR: required host path does not exist: $p" >&2
    exit 1
  fi
done


# Explicitly specify the full path to torchrun
export APPTAINERENV_TORCHRUN_PATH="/usr/local/bin/torchrun"

# Get the head node and its IP address
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
export APPTAINERENV_head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head Node: $head_node"
echo "Head Node IP: $APPTAINERENV_head_node_ip"

# Pass SLURM variables explicitly to the container
export APPTAINERENV_SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
export APPTAINERENV_SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="multinode.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

# Run distributed training inside the container
srun apptainer exec --nv \
  --bind $HOST_LIBFABRIC_LIB_PATH:/opt/libfabric/lib \
  --bind $HOST_LIBFABRIC_INCLUDE_PATH:/opt/libfabric/include \
  --bind $HOST_NCCL_LIB_PATH:/opt/nccl/lib \
  --bind $HOST_AWS_OFI_LIB_PATH:/opt/aws-ofi-nccl/lib \
  --bind $HOST_CXI_LIB_PATH:/usr/lib64 \
  --env LIBFABRIC_HOME=/opt/libfabric \
  --env NCCL_HOME=/opt/nccl \
  --env head_node_ip=$APPTAINERENV_head_node_ip \
  --env TRAINING_SCRIPT="$APPTAINERENV_TRAINING_SCRIPT" \
  --env TORCHRUN_PATH="$APPTAINERENV_TORCHRUN_PATH" \
  --env RDZV_ID=$SLURM_JOB_ID \
  --env SLURM_JOB_NUM_NODES=$APPTAINERENV_SLURM_JOB_NUM_NODES \
  --env SLURM_GPUS_ON_NODE=$APPTAINERENV_SLURM_GPUS_ON_NODE \
  $CONTAINER_PATH \
  bash -c 'export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/opt/nccl/lib:/usr/lib64:$LD_LIBRARY_PATH; \
  export CPATH=$LIBFABRIC_HOME/include:$CPATH; \
  $TORCHRUN_PATH \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$RDZV_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  $TRAINING_SCRIPT'

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"

```

## Key Changes from Multi-GPU to Multi-Node

The highlighted lines show the multi-node specific additions:

| Lines | Change | Purpose |
|-------|--------|---------|
| 8 | `--nodes=2` | Request multiple nodes |
| 10 | `--gpus-per-node=4` | GPUs per node (instead of `--gpus=4`) |
| 17 | `--batch-size 2048` | Larger batch for 8 GPUs |
| 20-22 | Module loading + NCCL version | Load NRIS GPU stack and NCCL 2.26.6 for container compatibility |
| 25-29 | Module-derived host paths | Derive bind sources from `EBROOT*` variables instead of hardcoded paths |
| 43-47 | Head node discovery | Get head node IP for rendezvous |
| 58-75 | `srun` with `--bind`/`--env` | Mount libfabric, NCCL, aws-ofi-nccl paths resolved from modules |
| 78-84 | `torchrun` with rendezvous | Use `rdzv_backend=c10d` and `rdzv_endpoint` for multi-node coordination |

```{note}
The key difference from single-node multi-GPU is the **rendezvous setup**. Single-node uses `--standalone`, while multi-node requires explicit coordination via `--rdzv_backend=c10d` and `--rdzv_endpoint` pointing to the head node.
```

The output of this job script is shown below:

```bash
Epoch 95/100 completed in 0.771 seconds
Validation Loss: 1.1998, Validation Accuracy: 0.7101
Epoch Throughput: 63787.926 images/second
Epoch 96/100 completed in 0.759 seconds
Validation Loss: 1.1924, Validation Accuracy: 0.7090
Epoch Throughput: 64736.418 images/second
Epoch 97/100 completed in 0.770 seconds
Validation Loss: 1.1911, Validation Accuracy: 0.7092
Epoch Throughput: 63812.132 images/second
Epoch 98/100 completed in 0.763 seconds
Validation Loss: 1.1671, Validation Accuracy: 0.7128
Epoch Throughput: 64432.161 images/second
Epoch 99/100 completed in 0.756 seconds
Validation Loss: 1.1799, Validation Accuracy: 0.7160
Epoch Throughput: 64995.126 images/second
Epoch 100/100 completed in 0.767 seconds
Validation Loss: 1.2086, Validation Accuracy: 0.7082
Epoch Throughput: 64118.564 images/second

Training Summary:
Total training time: 77.784 seconds
Throughput: 63190.172 images/second
Number of nodes: 2
Number of GPUs per node: 4
Total GPUs used: 8
Training completed successfully.
Stopping GPU utilization monitoring...
```

With 8 GPUs across 2 nodes, the throughput increased from ~5,100 images/second (single GPU) to ~63,000 images/second—a **12x speedup**. Training time dropped from ~16 minutes to just ~1.3 minutes.
