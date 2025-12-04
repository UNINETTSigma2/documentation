---
orphan: true
---

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

#!/bin/bash
#SBATCH --account=<project_number>
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


cd "${SLURM_SUBMIT_DIR}"



# Set the libfabric and nccl path from the host

HOST_LIBFABRIC_LIB_PATH=/opt/cray/libfabric/1.22.0/lib64
HOST_LIBFABRIC_INCLUDE_PATH=/opt/cray/libfabric/1.22.0/include
HOST_NCCL_PATH=/cluster/work/projects/nn9997k/software/nccl
HOST_NVIDIA_HPC_LIB_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/compilers/lib
HOST_CXI_LIB_PATH=/usr/lib64  # Directory containing libcxi.so.1


# Explicitly specify the full path to torchrun
export APPTAINERENV_TORCHRUN_PATH="/usr/local/bin/torchrun"

# Debugging: Enable NCCL logs
#export APPTAINERENV_NCCL_DEBUG=INFO
#export APPTAINERENV_NCCL_DEBUG_SUBSYS=ALL


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

#--pwd /cluster/work/projects/nn9997k/binod/PyTorch/private/simple_nn_project

# Run the training script with torchrun inside the container
srun  apptainer exec  --nv \
  --bind $HOST_LIBFABRIC_LIB_PATH:/opt/libfabric/lib \
  --bind $HOST_LIBFABRIC_INCLUDE_PATH:/opt/libfabric/include \
  --bind $HOST_NCCL_PATH:/opt/nccl \
  --bind $HOST_CXI_LIB_PATH:/usr/lib64 \
  --bind $HOST_NVIDIA_HPC_LIB_PATH:/opt/nvidia/hpc_sdk/lib \
  --env LIBFABRIC_HOME=/opt/libfabric \
  --env NCCL_HOME=/opt/nccl \
  --env NVIDIA_HPC_HOME=/opt/nvidia/hpc_sdk \
  --env head_node_ip=$APPTAINERENV_head_node_ip \
  --env TRAINING_SCRIPT="$APPTAINERENV_TRAINING_SCRIPT" \
  --env TORCHRUN_PATH="$APPTAINERENV_TORCHRUN_PATH" \
  --env RDZV_ID=$SLURM_JOB_ID \
  --env SLURM_JOB_NUM_NODES=$APPTAINERENV_SLURM_JOB_NUM_NODES \
  --env SLURM_GPUS_ON_NODE=$APPTAINERENV_SLURM_GPUS_ON_NODE \
  $CONTAINER_PATH \
  bash -c 'export LD_LIBRARY_PATH=$LIBFABRIC_HOME/lib:$NCCL_HOME/lib:$NVIDIA_HPC_HOME/lib:/usr/lib64:$LD_LIBRARY_PATH; \
  export CPATH=$LIBFABRIC_HOME/include:$CPATH; \
  $TORCHRUN_PATH  \
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

The output of this job script is shown below:

```bash
Epoch 95/100 completed in 0.770 seconds
Validation Loss: 1.3100, Validation Accuracy: 0.6904
Epoch Throughput: 63854.271 images/second
Epoch 96/100 completed in 0.773 seconds
Validation Loss: 1.4148, Validation Accuracy: 0.6721
Epoch Throughput: 63566.128 images/second
Epoch 97/100 completed in 0.776 seconds
Validation Loss: 1.4640, Validation Accuracy: 0.6532
Epoch Throughput: 63376.695 images/second
Epoch 98/100 completed in 0.777 seconds
Validation Loss: 1.4926, Validation Accuracy: 0.6459
Epoch Throughput: 63296.973 images/second
Epoch 99/100 completed in 0.766 seconds
Validation Loss: 1.4309, Validation Accuracy: 0.6559
Epoch Throughput: 64183.920 images/second
Epoch 100/100 completed in 0.784 seconds
Validation Loss: 1.4138, Validation Accuracy: 0.6648
Epoch Throughput: 62660.118 images/second

Training Summary:
Total training time: 78.706 seconds
Throughput: 62450.056 images/second
Number of nodes: 2
Number of GPUs per node: 4
Total GPUs used: 8
Training completed successfully.
Stopping GPU utilization monitoring...
```

The output demonstrates that using multiple nodes, each with four GPUs, significantly improves performance. The training time has decreased to `78.706 seconds`, and the throughput has increased from 2598.388 images/second in the single GPU setup to `62450.056 images/second`. This highlights the impressive optimization of Grace Hopper chips for efficient scaling of neural network training across multiple nodes.

If you find that the scaling does not meet your expectations for a specific use case, please feel free to reach out to us. We can assist in configuring environment variables recommended by HPE to optimize performance further. These details will also be covered in the BPG as previously discussed.
