(pytorch-single-gpu)=
# Single-GPU Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 1 of the PyTorch on Olivia guide. See {ref}`pytorch-on-olivia` for the overview, software choice, storage recommendations, and the full guide structure.

The goal of this part is to run the reference training workflow on a single GH200 GPU before scaling to multiple GPUs and multiple nodes.

## Learning Outcomes

By the end of this part, you can:

1. Run a PyTorch training job on **1 GPU** on Olivia.
2. Submit and monitor the job with Slurm.
3. Confirm success from expected log output.

```{note}
**Key considerations for Olivia:**
- The login node (x86_64) and GPU compute nodes (Aarch64) have different architectures. Software and containers must be built for ARM (Aarch64) to run on the compute nodes.
```
In order to be able to use PyTorch on Olivia we provide different solutions. You can read more about those solutions in detail here. ({ref}`access-pytorch`)


## Single GPU Implementation

```{warning}
Before moving forward, please make sure you clone the project repo, which you can find in this {ref}`pytorch-on-olivia` .
```


Once you clone the repo you will see the following scripts inside the project directory and we will discuss what those scripts does in brief.

```bash
olivia_pytorch
├── scripts
│   ├── train_utils.py
│   ├── train.py
│   ├── train_ddp.py
│   ├── model.py
│   ├── device_utils.py
│   └── dataset_utils.py
├── jobs
│   ├── singlegpu.sh
│   ├── multinode.sh
│   ├── multigpu.sh
│   └── logs
└── datasets
    └── download_datasets.sh
```
After cloning the repo, please download the required datasets by following these instructions.
```bash
cd datasets
chmod +x download_datasets.sh
./download_datasets.sh
```

To train the  model on a single GPU, we use the `train.py` file, which is the main training script. Moreover, when we perform training on multiple GPUs on a single node and on multiple nodes, we will use the `train_ddp.py` file.

Each of those scripts are discussed briefly below. You can click on the file if you want to see complete script in the github.
### [device_utils.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/device_utils.py)

The purpose of this script is to pick the compute device for training/ evaluation. It checks whether CUDA is available or not and if it is not available it still fallback to using CPU. This device object is used in training scripts to move model/data to the same hardware which is essential in deep learning.

### [model.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/model.py)

This module provides model architectures for comparative training and scaling experiments. The architecture are:
- a custom **WideResNet** (Convolutional Neural Network) 
-  **Vision Transformer (ViT)** wrapper

The script is designed to allow side-by-side performance benchmarking where we can compare throughput, memory footprint, convergence rates, and scaling behavior between traditional CNNs and transformer-based architectures from a same training pipeline.

WideResNet is a modular, three-stage CNN built with custom residual blocks and standard feature projection layers `ConvBnReLU`. This model is designed for image classification tasks and it progressively increases the channel depth while downsampling spatial dimensions. 

Whereas, class `ViTModel` is a transfer learning wrapper around `torchvision´s` pretrained ViT-B/16
architecture `vit_b_16`. It replaces, the default ImageNet classification head with a dense layer 
matching the target dataset´s class output.

### [dataset_utils.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/dataset_utils.py)

The `dataset_utils.py` module exposes entry points `load_cifar100()` and `load_imagenet()` to prepare training and evaluation data streams across single-GPU, multi-GPU, and multi-node setups.

To prevent cluster network proxy bottlenecks, SSL failures, and multi-rank download race conditions, the module assumes datasets are pre-fetched into `<repo_root>/datasets` via the lightweight `download_datasets.sh` utility script. For Tiny-ImageNet, `dataset_utils.py` automatically handles single-pass archive extraction and restructures validation images into PyTorch's standard ImageFolder class layout. In distributed environments `torchrun`, rank synchronization `dist.barrier()` ensures `Rank 0` performs disk operations exclusively while worker ranks wait, preventing filesystem collisions.

Finally, the module returns optimized PyTorch DataLoaders pre-configured with dataset-specific augmentations, normalization statistics, pin_memory acceleration, and optional DistributedSampler instances for scalable multi-GPU training.

### [train_utils.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/train_utils.py)

This module defines the core execution loops for single-GPU entry points. It abstracts model training and validation into two decoupled routines `train()` and `test()` ensuring consistent, mathematically exact metric collection and optimal GPU utilization.
 
The `train()` function executes single epoch training iterations. It manages non-blocking host-to-device tensor transfers `non_blocking=True`, forward/backward operations, and optimizer updates. To reduce Python execution overhead inside high-throughput iteration loops, Mixed Precision `torch.amp.autocast` and gradient scaling `GradScaler` eligibility are evaluated once at the header of the function rather than checked on every batch pass.

Similarly, the `test()` function evaluates model performance under validation constraints `model.eval()` and `torch.no_grad()`. It disables non-deterministic layers (such as Dropout) and skips gradient computation to minimize memory VRAM overhead during evaluation.

Moreover, to leverage Tensor Cores on  modern GPU architectures such as  NVIDIA GH200, it also automatically sets `torch.set_float32_matmul_precision("high")` when available.

```{note}
`train_utils.py` is reserved exclusively for single-GPU runs `train.py`. Distributed training `train_ddp.py` maintains its own local-rank loops to perform cross-GPU metric aggregation via `dist.all_reduce()`.
```

### [train.py](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/scripts/train.py)

The `train.py` script acts as the main controller for single-GPU benchmarking runs. It integrates the shared utilities described above into an end-to-end training pipeline.

During the environment setup, it parses command-line flags, sets seeds for deterministic execution, and resolves target compute hardware via `device_utils.get_device()`. After the device is selected, it configures DataLoaders by using `dataset_utils`, initializes either `WideResNet` or `ViTModel` by using `model.py`. Then, `build_optimizer()` automatically selects optimal hyperparameters based on the chosen model architecture. For WideResNet, it defaults to SGD with momentum `0.9` and weight decay `5e-4`. For Vision Transformers (ViT), it defaults to AdamW `lr=3e-4`, `weight_decay=1e-4`. Custom optimizer overrides `--optimizer sgd|adam|adamw` consistently retain architecture-appropriate regularization.

Moreover, it delegates epoch updates and validation checks to `train_utils.py`, while recording epoch wall-clock times, throughput (images/sec), and evaluation accuracy. It also includes, built-in early stopping based on target accuracy thresholds. Hence, all the metrics are now printed from this script.


## Job Script for Single GPU Training

The repo that you clone earlier already has the job script that use the [NRIS module](https://github.com/UNINETTSigma2/nris-tutorials/blob/main/pytorch-tutorial/jobs/singlegpu.sh) . Below you will find the job scripts using the container and EESSI stack to train the model based on your workflow.

`````{tabs}
````{group-tab} Container Implementation

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_singlegpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/singlegpu_%j.out
#SBATCH --error=logs/singlegpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=24           # Right-sized CPU allocation for ViT + Tiny-ImageNet
#SBATCH --mem=64G                    # Right-sized RAM for single-GPU ViT run
#SBATCH --gpus-per-node=1            # Request 1 GPU

# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to container and training script
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train.py --model vit --dataset tiny-imagenet --batch-size 256 --epochs 100 --optimizer adamw --base-lr 0.0003 --seed 42 --num-workers 4 --amp"

# Check GPU availability inside the container
echo "Checking GPU availability inside the container..."
apptainer exec --nv $CONTAINER_PATH python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/singlegpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run the single-GPU training script inside the container
apptainer exec --nv $CONTAINER_PATH python $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID
```

````

````{group-tab} EESSI Module

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=pytorch_singlegpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/singlegpu_%j.out
#SBATCH --error=logs/singlegpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=16           # Reserve 16 CPU cores (Right-sized for WideResNet + CIFAR-100)
#SBATCH --mem=48G                    # Request 48 GB RAM (Right-sized for WideResNet + CIFAR-100)
#SBATCH --gpus-per-node=1            # Request 1 GPU

module load EESSI/2025.06
module load torchvision/0.27.0-foss-2025b-PyTorch-2.12.0-CUDA-12.9.1


# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Path to training script
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train.py --model wideresnet --dataset cifar100 --seed 42 --batch-size 256 --epochs 100 --amp"


# Check GPU availability
echo "Checking GPU availability..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/singlegpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &
NVIDIA_MONITOR_PID=$!

# Run the training script
python $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill $NVIDIA_MONITOR_PID
```

````
`````

Then you can submit and monitor the running job using these commands:

`````{tabs}
````{group-tab} Submit & Monitor Commands

```bash
sbatch singlegpu.sh
squeue -u $USER
tail -f singlegpu_<jobid>.out
```

````

`````


Example output showing training progress:

```bash
Epoch 95/100: time=6.744s, train_loss=0.2986, train_acc=0.9116, val_loss=1.6322, val_acc=0.6370, throughput=7402.2 img/s
Epoch 96/100: time=6.727s, train_loss=0.3049, train_acc=0.9111, val_loss=1.9496, val_acc=0.5889, throughput=7420.6 img/s
Epoch 97/100: time=6.733s, train_loss=0.3189, train_acc=0.9050, val_loss=1.7844, val_acc=0.5987, throughput=7413.9 img/s
Epoch 98/100: time=6.717s, train_loss=0.3279, train_acc=0.9031, val_loss=1.6948, val_acc=0.6084, throughput=7431.8 img/s
Epoch 99/100: time=6.726s, train_loss=0.3269, train_acc=0.9034, val_loss=1.5346, val_acc=0.6322, throughput=7422.0 img/s
Epoch 100/100: time=6.734s, train_loss=0.3222, train_acc=0.9055, val_loss=1.7360, val_acc=0.6114, throughput=7412.8 img/s

Training complete. Final val_acc: 0.6114
Total time: 677.6s, final throughput: 7367.5 img/s
```

The output shows a throughput of approximately **7367 images/second** on a single GH200 GPU. In the next parts of this guide, we'll scale this up to multiple GPUs and see significant speedups.

Success criteria for Part 1:

- Job reaches `Training complete`
- Final summary prints a non-zero throughput in `img/s`
- No CUDA initialization errors in `.err` log


Now the goal is to scale this up to multiple GPUs. For this, please check out the {ref}`Multi GPU Guide <pytorch-multi-gpu>`.
