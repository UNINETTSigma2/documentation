(pytorch-multi-gpu)=
# Multi-GPU Implementation for PyTorch on Olivia

To scale our training to multiple GPUs, we will utilize PyTorch's Distributed Data Parallel (DDP) framework. DDP allows us to efficiently scale training across multiple GPUs and even across multiple nodes.To learn more about how to implement DDP, please refer to this official documentation from PyTorch [Getting Started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

For this, we need to modify the main Python script to include DDP implementation. The updated script will work for both scenarios Multiple GPUs within a single node and Multiple nodes.

Note: Please replace this path `/cluster/work/projects/<project_number>/<user_name>/olivia/datasets/` used in the script below to your actual path.

```python
#train_ddp.py
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

#Parse input arguments
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
    device = torch.device(f"cuda:{local_rank}")  #note: we haven´t used get_device from device_utils here

   # Log initialization info
    if global_rank == 0:
        print(f"Training started with {world_size} processes across {world_size // torch.cuda.device_count()} nodes.")
        print(f"Using {torch.cuda.device_count()} GPUs per node.")

    # Load the CIFAR-100 dataset with DistributedSampler
    per_gpu_batch_size = args.batch_size // world_size  # Divide global batch size across GPUs
    train_sampler = DistributedSampler(
        torchvision.datasets.CIFAR100(
            root="/cluster/work/projects/<project_number>/<user_name>/olivia/datasets/",
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
    scaler = torch.cuda.amp.GradScaler()
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
            with torch.cuda.amp.autocast():
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


## Key Changes and Addition for adapting single gpu setup to multi-gpu setup

1. Distributed Environment Setup

The single-GPU code uses get_device() from device_utils to determine the compute device (CPU or GPU).However, for the Multi-Gpu implementation, we introduced `ddp_setup()` to initialize the distributed environment using `torch.distributed.init_process_group` with the `NCCL` backend for GPU communication.We also set the local GPU for each processing using `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))`

2. Batch Size Adjustment

For training using the single GPU, the batch size is fixed and used directly for training whereas for the training using the the multiple GPU, the global batch size is divided across GPUs, `per_gpu_batch_size = args.batch_size // world_size` which ensures that each GPU processes an equal portion of the data.

3. Data Loading with Distributed Sampler

For the single-GPU implementation, the data loader is created without any special sampling mechanism. Howeve, for the multi-gpu setup we introduced `DistributedSampler` to ensure that each GPU processes a unique subset of the dataset.The sampler is updated at the start of the epoch using `train_sampler.set_epoch(epoch)`.

4. Model Wrapping with DDP
The model is directly moved to GPU using `.to(device)` in the single-gpu setup. However, for the multi-GPU setup,the model is wrapped with `torch.nn.parallel.DistributedDataParallel` to enable synchronized training across GPUs as shown here `model = DDP(model, device_ids=[local_rank])`

5. Mixed Precision Training
In the single-gpu setup the training is performed in full precision (FP32), whereas for the multi-gpu setup mixed precision training is introduced  using `torch.cuda.amp` for faster computation and reduced memory usage.The forward pass is wrapped in `torch.cuda.amp.autocast()` and Gradient scaling is handled using `torch.cuda.amp.GradScalar`

6. Synchronization Across GPUs
For the single-gpu usage no synchronization is required as only one GPU is used. However, for the multi-gpu setup we used `torch.distributed.barrier()` to synchronize all processes after each epoch.
Moroover, the validation metrics (accuracy and loss ) are averaged across all GPUs using `torch.distributed.all_reduce`

7. Distributed Environment Cleanup
Finally, we dont need to write any code for the cleanup for our single-gpu implementation.However, the distributed environment is cleaned up at the end of the training using `destroy_process_group()`


## Job Script for Multi GPU Training

To run the training on multiple GPUs, we can use the same job script mentioned earlier, but specify a higher number of GPUs.

When using `torchrun` for a single-node setup, you need to include the `--standalone` argument. However, this argument is not required for a multi-node setup. 
This job script is designed to train the model across multiple GPUs within a single node. In this script, we explicitly define the path to the torchrun executable using the following line:

`TORCHRUN_PATH="/usr/local/bin/torchrun"`

Note:If the torchrun executable is already in your `$PATH`, explicitly setting the path might not be necessary.



```bash
#!/bin/bash
#SBATCH --job-name=resnet_multigpu
#SBATCH --account=<project_number>
#SBATCH --output=multigpu_%j.out
#SBATCH --error=multigpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1             # Use one compute node
#SBATCH --ntasks-per-node=1   #  Single task per node
#SBATCH --cpus-per-task=72.   # Reserve 72 CPU cores(Eacho node has 256 CPUs)
#SBATCH --mem=440G            # Request 440 GB RAM for 4 GPUs(Each node has 768 GiB total )
#SBATCH --gpus=4              # Number of GPUs

# Path to the container
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"


# Path to the training script
TRAINING_SCRIPT="train_ddp.py --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2"

cd "${SLURM_SUBMIT_DIR}/.."



# Explicitly specify the full path to torchrun
TORCHRUN_PATH="/usr/local/bin/torchrun"

apptainer exec --nv $CONTAINER_PATH which torchrun
# Start GPU utilization monitoring in the background
GPU_LOG_FILE="multigpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

# Run the training script with torchrun inside the container
apptainer exec --nv  $CONTAINER_PATH $TORCHRUN_PATH --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"

```

Please note that, we increase the batch size and also the base learning rate compared to the single gpu implementation which are configurations specific to Neural Network training that we need to adjust when scaling to multiple GPUs. The output of the training is shown below:

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

Note that, By using four GPUs now we can see that the throughput is  `36926.594 images/second` and training time is `133.107 seconds`.It suggests that we achieved perfect scaling by using four GPUs.This setup is highly efficient, with a speedup factor of  almost `15.21` and a scaling efficiency of almost `374.1%`. This is the indication that  distributed training setup is highly optimized.

Once we are done with it, we can use the same python script where we had setup the ddp.We just need to make changes in the job script. Please refer to this  {ref}`Multi node Guide <pytorch-multi-node>` documentation to learn more about it.