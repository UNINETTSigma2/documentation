(pytorch-multi-node)=

# Multi-Node Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 3 of the PyTorch on Olivia guide. See {ref}`pytorch-single-gpu` for single-GPU and {ref}`pytorch-multi-gpu` for multi-GPU setup.

```{note}
The [ddp_train.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/train_ddp.py) file does not need any changes when scaling from single node to multiple nodes. The only change required is in the job script.
```

Multi-node training on Olivia requires a consistent NCCL-enabled module environment and a stable rendezvous endpoint shared by all nodes. The job script below handles both.

## Learning Outcomes

By the end of this part, you can:

1. Launch PyTorch training across **multiple nodes** with `torchrun`.
2. Configure the required module environment for distributed communication.
3. Set rendezvous parameters correctly for a stable multi-node start.



## Job Script for Multi-Node Training

The repo which you cloned earlier already has the job script that uses the [NRIS module](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/jobs/multinode.sh). Below, you will find the equivalent job scripts for container and EESSI 
stack.

`````{tabs}

````{group-tab} Container Implementation

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_multinode
#SBATCH --account=<project_number>
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=2                    # Request 2 compute nodes
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=48           # Right-sized CPU allocation for multi-node ViT DDP
#SBATCH --mem=192G                   # Right-sized RAM per node for multi-node ViT DDP
#SBATCH --gpus-per-node=4            # Number of GPUs per node

module load NRIS/GPU     
module load aws-ofi-nccl/1.19.1-GCCcore-14.3.0-CUDA-13.0.0
module load libfabric/2.3.1-GCCcore-14.3.0-CUDA-13.0.0

# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to container and training script
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"

TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model vit --dataset tiny-imagenet --epochs 100 --batch-size 2048 --optimizer adamw --base-lr 0.00015 --target-accuracy 0.95 --patience 2 --seed 42 --num-workers 8 --amp"


# Host library paths
HOST_LIBFABRIC_LIB="${EBROOTLIBFABRIC}/lib"
HOST_AWSOFI_LIB="${EBROOTAWSMINOFIMINNCCL}/lib"
HOST_CXI_LIB_PATH="/usr/lib64"

# NCCL debug
#export APPTAINERENV_NCCL_DEBUG=INFO
#export APPTAINERENV_NCCL_DEBUG_SUBSYS=INIT,NET


# Get head node IP
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
export APPTAINERENV_head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Head Node: $head_node"
echo "Head Node IP: $APPTAINERENV_head_node_ip"

# Pass SLURM variables explicitly
export APPTAINERENV_SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES
export APPTAINERENV_SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE

# Start GPU utilization monitoring
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/multinode.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run training script with torchrun inside container
srun apptainer exec --nv \
  --bind $HOST_LIBFABRIC_LIB:/opt/libfabric/lib \
  --bind $HOST_AWSOFI_LIB:/opt/aws-ofi-nccl/lib \
  --bind $HOST_CXI_LIB_PATH:/usr/lib64 \
  --env head_node_ip=$APPTAINERENV_head_node_ip \
  --env TRAINING_SCRIPT="$TRAINING_SCRIPT" \
  --env RDZV_ID=$SLURM_JOB_ID \
  --env SLURM_JOB_NUM_NODES=$APPTAINERENV_SLURM_JOB_NUM_NODES \
  --env SLURM_GPUS_ON_NODE=$APPTAINERENV_SLURM_GPUS_ON_NODE \
  $CONTAINER_PATH \
  bash -c 'export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/libfabric/lib:/usr/lib64:$LD_LIBRARY_PATH; \
  torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$RDZV_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  $TRAINING_SCRIPT'

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID 2>/dev/null || true
```

````

````{group-tab} EESSI Module

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_multinode
#SBATCH --account=<project_number>
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=2                    # Request 2 compute nodes
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=40           # Reserve 40 CPU cores (Right-sized for multi-node WideResNet)
#SBATCH --mem=128G                   # Request 128 GB RAM (Right-sized for multi-node WideResNet)
#SBATCH --gpus-per-node=4            # Number of GPUs per node

# Activate the EESSI environment
module load EESSI/2025.06
module load torchvision/0.27.0-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1


# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to the training script
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model wideresnet --dataset cifar100 --epochs 100 --batch-size 2048 --base-lr 0.02 --target-accuracy 0.95 --patience 2 --seed 42 --amp"


# NCCL Debug
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET


# Get head node IP
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"


# Start GPU utilization monitoring
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/multinode.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run training script with torchrun inside container
srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  $TRAINING_SCRIPT

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID 2>/dev/null || true
```

````
`````

Then you can submit and monitor the running job using these commands:

`````{tabs}
````{group-tab} Submit & Monitor Command

```bash
sbatch multinode.sh
squeue -u $USER
tail -f multinode_<jobid>.out
```

````

````
`````

## Key Changes from Multi-GPU to Multi-Node

The multi-node-specific additions are:

| Change | Purpose |
|-------|---------|
| `#SBATCH --nodes=2` and `#SBATCH --gpus-per-node=4` | Requests resources on multiple nodes |
| Head-node hostname from `SLURM_JOB_NODELIST` | Defines rendezvous endpoint for all processes |
| `srun torchrun ... --rdzv_backend=c10d --rdzv_endpoint=...` | Coordinates multi-node process-group formation |

```{note}
The key difference from single-node multi-GPU is the **rendezvous setup**. Single-node uses `--standalone`, while multi-node requires explicit coordination via `--rdzv_backend=c10d` and `--rdzv_endpoint` pointing to the head node.
```

The output of this job script is shown below:

```bash
Epoch 95/100: time=1.271s, train_loss=0.0146, train_acc=0.9994, val_loss=1.4960, val_acc=0.6831, throughput=38657.3 img/s
Epoch 96/100: time=1.271s, train_loss=0.0154, train_acc=0.9990, val_loss=1.5086, val_acc=0.6815, throughput=38674.0 img/s
Epoch 97/100: time=1.290s, train_loss=0.0164, train_acc=0.9987, val_loss=1.4687, val_acc=0.6835, throughput=38103.6 img/s
Epoch 98/100: time=1.282s, train_loss=0.0168, train_acc=0.9991, val_loss=1.4859, val_acc=0.6829, throughput=38335.4 img/s
Epoch 99/100: time=1.315s, train_loss=0.0143, train_acc=0.9994, val_loss=1.4213, val_acc=0.6907, throughput=37383.9 img/s
Epoch 100/100: time=1.270s, train_loss=0.0131, train_acc=0.9994, val_loss=1.3783, val_acc=0.6962, throughput=38694.6 img/s

Training Summary:
Total training time: 131.793 seconds
Throughput: 37294.946 images/second
Total GPUs used: 8
Training completed successfully.
```

With 8 GPUs across 2 nodes, the throughput increased from ~7367  images/second (single GPU) to ~37294 images/second—a **5x speedup**. This sub-linear scaling achieves roughly `63%` scaling efficiency across 8 GPUs, where multi-node communication overhead, specifically inter-node network latency during gradient synchronization across nodes is preventing linear scaling. Moreover, the training time dropped from `~667` seconds to just `~131` seconds.

### Success criteria for Part 3:

- Log shows `Head Node` and a resolved head-node IP
- Final summary reports `Number of nodes: 2` and `Total GPUs used: 8`
- Training completes without rendezvous or NCCL startup errors

