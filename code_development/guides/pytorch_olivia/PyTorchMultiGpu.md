(pytorch-multi-gpu)=
# Multi-GPU Implementation for PyTorch on Olivia

```{contents}
:depth: 2
```

This is part 2 of the PyTorch on Olivia guide. See {ref}`pytorch-on-olivia` for the single-GPU setup.

To scale training across multiple GPUs, we use PyTorch's [Distributed Data Parallel (DDP)](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html). The code below works for both single-node multi-GPU and multi-node configurations.

```{code-block} python
:linenos:
:emphasize-lines: 8-10, 28-31, 34, 48-49, 64-65, 71-72, 79, 91-93, 101, 113-116, 143

# train_ddp.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dataset_utils import load_cifar100
from model import WideResNet
from train_utils import test

# Configuration
DATA_DIR = "./datasets"  # Dataset downloads automatically here

# Parse input arguments
parser = argparse.ArgumentParser(description='CIFAR-100 DDP example with Mixed Precision',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=512, help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01, help='Learning rate for single GPU')
parser.add_argument('--target-accuracy', type=float, default=0.85, help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2, help='Number of epochs that meet target before stopping')
args = parser.parse_args()

def ddp_setup():
    """Set up the distributed environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main_worker():
    ddp_setup()

   # Get the local rank and device
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")  # Note: we don't use get_device from device_utils here

   # Log initialization info
    if global_rank == 0:
        print(f"Training started with {world_size} processes across {world_size // torch.cuda.device_count()} nodes.")
        print(f"Using {torch.cuda.device_count()} GPUs per node.")

    # Load the CIFAR-100 dataset with DistributedSampler
    per_gpu_batch_size = args.batch_size // world_size  # Divide global batch size across GPUs
    train_sampler = DistributedSampler(
        torchvision.datasets.CIFAR100(
            root=DATA_DIR,
            train=True,
            download=True
        )
    )
    train_loader, test_loader = load_cifar100(
        batch_size=per_gpu_batch_size,
        num_workers=8,
        sampler=train_sampler
    )

    # Create the model and wrap it with DDP
    num_classes = 100  # CIFAR-100 has 100 classes
    model = WideResNet(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=5e-4)

    # Initialize gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')
    val_accuracy = []
    total_time = 0
    total_images = 0  # Total images processed globally

    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # Set the sampler epoch for shuffling
        model.train()
        t0 = time.time()

       # Train the model for one epoch
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

           # Zero the gradients
            optimizer.zero_grad()

           # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

           # Backward pass and optimization with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Synchronize all processes
        torch.distributed.barrier()
        epoch_time = time.time() - t0
        total_time += epoch_time

        # Compute throughput (images per second for this epoch)
        images_per_sec = len(train_loader) * args.batch_size / epoch_time
        total_images += len(train_loader) * args.batch_size

        # Compute validation accuracy and loss
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)

        # Average validation metrics across all GPUs
        v_accuracy_tensor = torch.tensor(v_accuracy).to(device)
        v_loss_tensor = torch.tensor(v_loss).to(device)
        torch.distributed.all_reduce(v_accuracy_tensor, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(v_loss_tensor, op=torch.distributed.ReduceOp.AVG)

        # Print metrics only from the main process
        if global_rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.3f} seconds")
            print(f"Validation Loss: {v_loss_tensor.item():.4f}, Validation Accuracy: {v_accuracy_tensor.item():.4f}")
            print(f"Epoch Throughput: {images_per_sec:.3f} images/second")

        # Early stopping
        val_accuracy.append(v_accuracy_tensor.item())
        if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
            if global_rank == 0:
                print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break

    # Log total training time and summary
    if global_rank == 0:
        throughput = total_images / total_time
        print("\nTraining Summary:")
        print(f"Total training time: {total_time:.3f} seconds")
        print(f"Throughput: {throughput:.3f} images/second")
        print(f"Number of nodes: {world_size // torch.cuda.device_count()}")
        print(f"Number of GPUs per node: {torch.cuda.device_count()}")
        print(f"Total GPUs used: {world_size}")
        print("Training completed successfully.")

    # Clean up the distributed environment
    destroy_process_group()
if __name__ == '__main__':
    main_worker()
```


## Key Changes from Single-GPU to Multi-GPU

The highlighted lines above show the DDP-specific additions:

| Lines | Change | Purpose |
|-------|--------|---------|
| 8-10 | DDP imports | `DistributedSampler`, `DDP`, `init_process_group` |
| 28-31 | `ddp_setup()` | Initialize NCCL backend and set local GPU |
| 34 | Call `ddp_setup()` | Start distributed environment |
| 48-49 | Batch size division | Split global batch across GPUs |
| 64-65 | Wrap model with `DDP` | Enable synchronized gradient updates |
| 71-72 | Mixed precision setup | `GradScaler` for FP16 training |
| 79 | `set_epoch()` | Ensure proper shuffling across epochs |
| 91-93 | `autocast()` context | Run forward pass in FP16 |
| 101 | `barrier()` | Synchronize all processes after epoch |
| 113-116 | `all_reduce()` | Average metrics across GPUs |
| 143 | `destroy_process_group()` | Clean up distributed environment |


## Job Script for Multi-GPU Training


For single-node multi-GPU training, use `torchrun` with `--standalone`. We request 4 GPUs and adjust batch size and learning rate for better scaling.

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=resnet_multigpu
#SBATCH --account=<project_number>
#SBATCH --output=multigpu_%j.out
#SBATCH --error=multigpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --gpus=4

CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"

# Run training with 4 GPUs
apptainer exec --nv $CONTAINER_PATH torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_ddp.py --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2
```

Example output:

```bash
Epoch 95/100 completed in 1.344 seconds
Validation Loss: 1.4165, Validation Accuracy: 0.6663
Epoch Throughput: 36577.311 images/second
Epoch 96/100 completed in 1.312 seconds
Validation Loss: 1.3927, Validation Accuracy: 0.6686
Epoch Throughput: 37470.264 images/second
Epoch 97/100 completed in 1.315 seconds
Validation Loss: 1.3615, Validation Accuracy: 0.6757
Epoch Throughput: 37378.521 images/second
Epoch 98/100 completed in 1.351 seconds
Validation Loss: 1.2696, Validation Accuracy: 0.7011
Epoch Throughput: 36380.679 images/second
Epoch 99/100 completed in 1.299 seconds
Validation Loss: 1.4111, Validation Accuracy: 0.6724
Epoch Throughput: 37825.847 images/second
Epoch 100/100 completed in 1.356 seconds
Validation Loss: 1.3142, Validation Accuracy: 0.6903
Epoch Throughput: 36251.588 images/second

Training Summary:
Total training time: 133.107 seconds
Throughput: 36926.594 images/second
Number of nodes: 1
Number of GPUs per node: 4
Total GPUs used: 4
Training completed successfully.
```

With 4 GPUs, the throughput increased from ~2,600 images/second (single GPU) to ~37,000 images/second—a **14x speedup**. This super-linear scaling (beyond the expected 4x) comes from mixed precision training (FP16) and the larger effective batch size, which better utilizes the GPU compute capabilities.

To scale beyond a single node, see the {ref}`Multi-Node Guide <pytorch-multi-node>`.
