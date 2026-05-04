(pytorch-multi-node)=

# Multi-Node Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 3 of the PyTorch on Olivia guide. See {ref}`pytorch-single-gpu` for single-GPU and {ref}`pytorch-multi-gpu` for multi-GPU setup.

Multi-node training on Olivia requires a consistent NCCL-enabled module environment and a stable rendezvous endpoint shared by all nodes. The job script below handles both.

## Learning Outcomes

By the end of this part, you can:

1. Launch PyTorch training across **multiple nodes** with `torchrun`.
2. Configure the required module environment for distributed communication.
3. Set rendezvous parameters correctly for a stable multi-node start.



## Job Script for Multi-Node Training

Choose either module-based launch or direct container launch.

`````{tabs}
````{group-tab} Module Path

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --job-name=resnet_multinode_mod
#SBATCH --output=multinode_module_%j.out
#SBATCH --error=multinode_module_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G

set -euo pipefail

SCRIPT_DIR="/cluster/work/projects/<project_number>/<username>/pytorch_olivia"

ml reset
ml load NRIS/GPU
ml load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0
ml use /cluster/work/support/pytorch_module
ml load PyTorch/2.8.0

export PYTORCH_OVERLAY_MODE=ro

cd "${SCRIPT_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes[0]}"
export RDZV_ENDPOINT="${head_node}:29500"

echo "Head node: ${head_node}"
echo "Rendezvous endpoint: ${RDZV_ENDPOINT}"

srun torchrun \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train_ddp.py --epochs 100 --batch-size 2048 --base-lr 0.04 --target-accuracy 0.95 --patience 2

```

````

````{group-tab} Direct Container Path

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --job-name=resnet_multinode_ctr
#SBATCH --output=multinode_container_%j.out
#SBATCH --error=multinode_container_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G

set -euo pipefail

CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"
SCRIPT_DIR="/cluster/work/projects/<project_number>/<username>/pytorch_olivia"
TRAINING_SCRIPT="train_ddp.py --epochs 100 --batch-size 2048 --base-lr 0.04 --target-accuracy 0.95 --patience 2"

ml reset
ml load NRIS/GPU
ml load NCCL/2.26.6-GCCcore-14.2.0-CUDA-12.8.0

LIBFABRIC_LIB_PATH="${EBROOTLIBFABRIC}/lib"
LIBFABRIC_INCLUDE_PATH="${EBROOTLIBFABRIC}/include"
NCCL_ROOT_PATH="${EBROOTNCCL}"
AWS_OFI_NCCL_LIB_PATH="${EBROOTAWSMINOFIMINNCCL}/lib"
CXI_LIB_PATH="/usr/lib64"

HF_ROOT="${SCRIPT_DIR}/hf_cache"
mkdir -p "${HF_ROOT}/hub" "${HF_ROOT}/datasets" "${HF_ROOT}/torch"

cd "${SCRIPT_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address | awk '{print $1}')
rdzv_endpoint="${head_node_ip}:29500"

echo "Head node: ${head_node}"
echo "Head node IP: ${head_node_ip}"
echo "Rendezvous endpoint: ${rdzv_endpoint}"

srun apptainer exec --nv \
  --bind "${SCRIPT_DIR}:${SCRIPT_DIR}" \
  --bind "${LIBFABRIC_LIB_PATH}:/opt/libfabric/lib" \
  --bind "${LIBFABRIC_INCLUDE_PATH}:/opt/libfabric/include" \
  --bind "${NCCL_ROOT_PATH}:/opt/nccl" \
  --bind "${AWS_OFI_NCCL_LIB_PATH}:/opt/aws-ofi-nccl/lib" \
  --bind "${CXI_LIB_PATH}:${CXI_LIB_PATH}" \
  --pwd "${SCRIPT_DIR}" \
  --env FI_PROVIDER="${FI_PROVIDER:-cxi}" \
  --env FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}" \
  --env NCCL_PROTO="${NCCL_PROTO:-^LL128}" \
  --env LIBFABRIC_HOME="/opt/libfabric" \
  --env NCCL_HOME="/opt/nccl" \
  --env AWS_OFI_NCCL_HOME="/opt/aws-ofi-nccl" \
  --env RDZV_ENDPOINT="${rdzv_endpoint}" \
  --env RDZV_ID="${SLURM_JOB_ID}" \
  --env SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES}" \
  --env SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE}" \
  --env TRAINING_SCRIPT="${TRAINING_SCRIPT}" \
  --env HF_HOME="${HF_ROOT}" \
  --env HF_HUB_CACHE="${HF_ROOT}/hub" \
  --env HF_DATASETS_CACHE="${HF_ROOT}/datasets" \
  --env TRANSFORMERS_CACHE="${HF_ROOT}/hub" \
  --env TORCH_HOME="${HF_ROOT}/torch" \
  "${CONTAINER_PATH}" \
  bash -lc 'export LD_LIBRARY_PATH="${LIBFABRIC_HOME}/lib:${NCCL_HOME}/lib:${AWS_OFI_NCCL_HOME}/lib:/usr/lib64:${LD_LIBRARY_PATH}"; export CPATH="${LIBFABRIC_HOME}/include:${CPATH:-}"; torchrun --nnodes="${SLURM_JOB_NUM_NODES}" --nproc_per_node="${SLURM_GPUS_ON_NODE}" --rdzv_id="${RDZV_ID}" --rdzv_backend=c10d --rdzv_endpoint="${RDZV_ENDPOINT}" ${TRAINING_SCRIPT}'
```

````

````{group-tab} EESSI Path

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --account=<project_number>
#SBATCH --job-name=resnet_multinode_eessi
#SBATCH --output=multinode_eessi_%j.out
#SBATCH --error=multinode_eessi_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G

set -euo pipefail

SCRIPT_DIR="/cluster/work/projects/<project_number>/<username>/pytorch_olivia"

ml reset
module load EESSI/2025.06
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load torchvision/0.22.0-foss-2024a-CUDA-12.6.0

cd "${SCRIPT_DIR}"

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes[0]}"
export RDZV_ENDPOINT="${head_node}:29500"

echo "Head node: ${head_node}"
echo "Rendezvous endpoint: ${RDZV_ENDPOINT}"

srun torchrun \
  --nnodes="${SLURM_JOB_NUM_NODES}" \
  --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
  --rdzv_id="${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train_ddp.py --epochs 100 --batch-size 2048 --base-lr 0.04 --target-accuracy 0.95 --patience 2
```

````
`````

The submit and monitor commands are identical for both launch modes.

`````{tabs}
````{group-tab} Module Path

```bash
sbatch multinode_module.sh
squeue -u $USER
tail -f multinode_module_<jobid>.out
```

````

````{group-tab} Direct Container Path

```bash
sbatch multinode_container.sh
squeue -u $USER
tail -f multinode_container_<jobid>.out
```

````

````{group-tab} EESSI Path

```bash
sbatch multinode_eessi.sh
squeue -u $USER
tail -f multinode_eessi_<jobid>.out
```

````
`````

## Key Changes from Multi-GPU to Multi-Node

The multi-node-specific additions are:

| Change | Purpose |
|-------|---------|
| `#SBATCH --nodes=2` and `#SBATCH --gpus-per-node=4` | Requests resources on multiple nodes |
| `ml load NRIS/GPU` and `ml load NCCL/2.26.6-...` | Loads the distributed communication stack |
| Head-node hostname from `SLURM_JOB_NODELIST` | Defines rendezvous endpoint for all processes |
| `srun torchrun ... --rdzv_backend=c10d --rdzv_endpoint=...` | Coordinates multi-node process-group formation |

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
```

With 8 GPUs across 2 nodes, the throughput increased from ~5,100 images/second (single GPU) to ~63,000 images/second—a **12x speedup**. Training time dropped from ~16 minutes to just ~1.3 minutes.

Success criteria for Part 3:

- Log shows `Head Node` and a resolved head-node IP
- Final summary reports `Number of nodes: 2` and `Total GPUs used: 8`
- Training completes without rendezvous or NCCL startup errors

## Advanced Troubleshooting (Optional)

Use these only when debugging startup/performance issues:

```bash
# NCCL debug logs
export APPTAINERENV_NCCL_DEBUG=INFO
export APPTAINERENV_NCCL_DEBUG_SUBSYS=ALL

# Optional GPU utilization logging
GPU_LOG_FILE="multinode.log"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
```
