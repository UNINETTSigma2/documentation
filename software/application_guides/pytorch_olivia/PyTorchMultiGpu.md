(pytorch-multi-gpu)=
# Multi-GPU Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 2 of the PyTorch on Olivia guide. See {ref}`pytorch-single-gpu` for the single-GPU setup.

## Learning Outcomes

By the end of this part, you can:

1. Run the same training workflow on **4 GPUs on one node**.
2. Understand the minimum DDP changes from the single-GPU version.
3. Validate that distributed training launched correctly.

To scale training across multiple GPUs, we use PyTorch's [Distributed Data Parallel (DDP)](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html). The `train_ddp.py`code  works for both single-node multi-GPU and multi-node configurations. However, it is important to note that, we **don´t use** `train_utils.py` and `device_utils.py` which we discussed earlier in this page {ref}`pytorch-single-gpu` , as `train_ddp.py` is self contained for DDP and implements equivalent logic directly.


## Explanation  [train_ddp.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/train_ddp.py) file

The `train_ddp.py` script extends the training pipeline across multiple GPUs and nodes using PyTorch's `DistributedDataParallel` (DDP). It coordinates distributed environment setup, multi-process data distribution, and synchronized cross-rank metric collection while strictly adhering to PyTorch distributed training best practices.

It relies on launch utilities like `torchrun` to set process environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) and uses the `nccl` backend for high-speed inter-GPU communications.

### Key Implementation Details & PyTorch DDP Best Practices

1. **Process Group Initialization (`ddp_setup`)**: 

Reads environment variables dynamically (RANK, LOCAL_RANK, WORLD_SIZE), binds the process to its explicit local CUDA device, and initializes the process group using the NCCL backend.

2. **Rank 0 Synchronization & Race Prevention**:

In a distributed setup, managing filesystem access and logging from a single lead process is critical:

- Dataset Downloads: To avoid multi-process race conditions on shared cluster storage, `rank 0` performs archive extraction and directory setup exclusively while worker ranks pause at a `dist.barrier()` synchronization point in `dataset_utils.py`.

- Logging & Early Stopping: Epoch metrics and configuration details are logged exclusively from `rank 0` to prevent redundant `stdout` noise across nodes. When `rank 0` triggers an early stopping condition, it communicates this decision to all worker ranks via `dist.broadcast()` so every process exits synchronously.

3. **Data Sharding (`DistributedSampler`)**:

Divides the global batch size evenly across all active GPUs `per_gpu_batch_size = global_batch_size // world_size`. This allows scaling the total global batch size seamlessly in Slurm job scripts as node counts increase. Additionally, calling `train_sampler.set_epoch(epoch)` at the start of every epoch guarantees distinct, non-overlapping dataset shuffles across nodes.


4. **Exact Metric Aggregation (`all_reduce_metrics`)**: 

Avoids simple, imprecise averaging of local worker accuracies. Instead, each process tracks raw local sums (`correct_count`, `loss_sum`, `total_samples`) and aggregates them across the cluster using `dist.all_reduce(op=dist.ReduceOp.SUM)`. This produces mathematically exact global loss and accuracy metrics regardless of dataset splitting.

5. **Distributed Throughput Measurement**: 

Calculates epoch execution times using `dist.all_reduce(op=dist.ReduceOp.MAX)` across all GPUs. Measuring total processed images against the slowest worker rank provides a true reflection of synchronized step duration across the cluster.


6. **Rank-Specific Seeding & Cleanup**: 

- Seeding: Seeds random number generators with a global rank offset `seed + rank`, ensuring worker streams apply independent data augmentations while maintaining overall experiment reproducibility.

- Cleanup: Enforces process group termination `dist.destroy_process_group()` inside a finally block. This guarantees clean execution teardown and prevents orphaned CUDA processes from hanging cluster resources when scaling across nodes.


## Job Script for Multi-GPU Training


For single-node multi-GPU training, we should use `torchrun` command with `--standalone`. The repo you cloned earlier has the job script where you use the [NRIS module](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/jobs/multigpu.sh). If you choose to use container or EESSI stack the job scripts are given below.

`````{tabs}

````{group-tab} Container Implementation

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_multigpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/multigpu_%j.out
#SBATCH --error=logs/multigpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=48           # Right-sized CPU allocation for 4-GPU ViT DDP
#SBATCH --mem=192G                   # Right-sized RAM for 4-GPU ViT DDP
#SBATCH --gpus=4                     # Request 4 GPU

# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to container and training script
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"

TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model vit --dataset tiny-imagenet --batch-size 1024 --epochs 100 --optimizer adamw --base-lr 0.0003 --target-accuracy 0.95 --patience 2 --seed 42 --num-workers 8 --amp"



# Check GPU availability inside the container
echo "Checking GPU availability inside the container..."
apptainer exec --nv $CONTAINER_PATH python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/multigpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run the training script with torchrun inside the container
apptainer exec --nv $CONTAINER_PATH torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID
```

````

````{group-tab} EESSI Module

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_multigpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/multigpu_%j.out
#SBATCH --error=logs/multigpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=40           # Reserve 40 CPU cores (Right-sized for 4-GPU WideResNet)
#SBATCH --mem=128G                   # Request 128 GB RAM (Right-sized for 4-GPU WideResNet)
#SBATCH --gpus=4                     # Request 4 GPU

module load EESSI/2025.06
module load torchvision/0.27.0-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1

# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to the training script
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model wideresnet --dataset cifar100 --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2 --seed 42 --amp"

# Change working directory to project root
cd "${PROJECT_DIR}"

# Check GPU availability
echo "Checking GPU availability inside the container..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/multigpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run the training script with torchrun inside the container
torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID
```

````
`````

Then you can submit and monitor the running job using these commands:

`````{tabs}
````{group-tab} Submit & Monitor Command

```bash
sbatch multigpu.sh
squeue -u $USER
tail -f multigpu_<jobid>.out
```
````

````
`````

Example output:

```bash
Epoch 95/100: time=2.000s, train_loss=0.0077, train_acc=0.9997, val_loss=1.0301, val_acc=0.7474, throughput=24572.7 img/s
Epoch 96/100: time=1.986s, train_loss=0.0078, train_acc=0.9997, val_loss=1.0099, val_acc=0.7461, throughput=24753.2 img/s
Epoch 97/100: time=1.966s, train_loss=0.0095, train_acc=0.9995, val_loss=1.0698, val_acc=0.7380, throughput=24994.7 img/s
Epoch 98/100: time=1.975s, train_loss=0.0090, train_acc=0.9996, val_loss=1.0357, val_acc=0.7513, throughput=24891.3 img/s
Epoch 99/100: time=2.012s, train_loss=0.0081, train_acc=0.9997, val_loss=0.9978, val_acc=0.7515, throughput=24435.0 img/s
Epoch 100/100: time=1.984s, train_loss=0.0082, train_acc=0.9997, val_loss=1.0068, val_acc=0.7491, throughput=24777.5 img/s

Training Summary:
Total training time: 201.998 seconds
Throughput: 24332.868 images/second
Total GPUs used: 4
Training completed successfully.
```

With 4 GPUs and FP16 mixed precision, the throughput increased from ~7367  images/second (single GPU) to ~24,000 images/second—a **3x speedup**. This near-linear scaling demonstrates efficient distributed data parallelism, with slight overhead due to inter-GPU gradient communication.

### Success criteria for Part 2:

- Output includes `Training started with 4 processes`
- Final summary reports `Total GPUs used: 4`
- Throughput is substantially higher than Part 1

For Part 3 (multi-node), you keep the same `train_ddp.py` and only change the job launch configuration. See {ref}`Multi-Node Guide <pytorch-multi-node>`.
