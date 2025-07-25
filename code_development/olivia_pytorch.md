(pytorch-wheels)=

# PyTorch on Olivia

In this guide, we’ll be testing PyTorch on the Olivia system, which uses the Aarch64 architecture on its compute nodes. To do this, we’ll use specific PyTorch wheels compatible with this architecture.

Note: This documentation is a work in progress and is intended for testing purposes. If you encounter any issues or something does not work as expected, please let us know. Furthermore, this approach relies on `pip` installation, which can degrade the file system. As a result, it will no longer be permitted after the pilot phase.

Key Considerations:

1. Different Architectures:

The login node and the compute node on Olivia have different architectures. The login node uses the x86_64 architecture, while the compute node uses Aarch64. This means we cannot install software directly on the login node and expect it to work on the compute node.

2. Internet Connectivity on Compute Nodes:

At the time of testing, the compute nodes did not have direct internet access. Consequently, we had to install the required PyTorch wheels within a virtual environment using a job script. This ensured that the installation occurred on the compute node during the execution of the job script.

However, it is now possible to access the internet directly from the compute nodes by configuring proxies, as demonstrated below:

````bash
export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
````

3. CUDA Version:

The compute nodes are equipped with CUDA Version 12.7, as confirmed by running the nvidia-smi command. Therefore, we need to ensure that the PyTorch wheels we use are compatible with this CUDA version.



You can download the necessary PyTorch wheels for your project from the PyTorch nightly builds using the following link:

[torch](https://download.pytorch.org/whl/nightly/torch)

[pytorch-triton](https://download.pytorch.org/whl/nightly/pytorch-triton/)

[torchvision](https://download.pytorch.org/whl/nightly/torchvision/)

## Training a ResNet Model with the Fashion-MNIST Dataset

To test Olivia's capabilities with real-world workloads, we will train a ResNet model using the Fashion-MNIST dataset. The testing will be conducted under the following scenarios:

1. Single GPU

2. Multiple GPUs

3. Multiple Nodes

The primary goal of this exercise is to verify that we can successfully run training tasks on Olivia. As such, we will not delve into the specifics of neural network training in this documentation. A separate guide will be prepared to cover those details.



### Single GPU Implementation

To train the ResNet model on a single GPU, we used the following files. These files include the main Python script responsible for training the ResNet model.

````python
# resnet.py
"""
This script trains a WideResNet model on the Fashion MNIST dataset without using Distributed Data Parallel (DDP).
The goal is to provide a simple, single-GPU implementation of the training process.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
# Import custom modules
from data.dataset_utils import load_fashion_mnist
from models.wide_resnet import WideResNet
from training.train_utils import train, test
from utils.device_utils import get_device
# Define paths
shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared"))
images_dir = os.path.join(shared_dir, "images")
os.makedirs(images_dir, exist_ok=True)
# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.01
TARGET_ACCURACY = 0.85
PATIENCE = 2
def train_resnet_without_ddp(batch_size, epochs, learning_rate, device):
    """
    Trains a WideResNet model on the Fashion MNIST dataset without DDP.
    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
    Returns:
        None
    """
    print(f"Training WideResNet on Fashion MNIST with Batch Size: {batch_size}")
    # Training variables
    val_accuracy = []
    total_time = 0
    # Load the dataset
    train_loader, test_loader = load_fashion_mnist(batch_size=batch_size)
    # Initialize the WideResNet Model
    num_classes = 10
    model = WideResNet(num_classes).to(device)
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        t0 = time.time()
        # Train the model for one epoch
        train(model, optimizer, train_loader, loss_fn, device)
        # Calculate epoch time
        epoch_time = time.time() - t0
        total_time += epoch_time
        # Compute throughput (images per second)
        images_per_sec = len(train_loader) * batch_size / epoch_time
        # Compute validation accuracy and loss
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        val_accuracy.append(v_accuracy)
        # Print metrics
        print("Epoch = {:2d}: Epoch Time = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}, Images/sec = {:5.3f}, Cumulative Time = {:5.3f}".format(
            epoch + 1, epoch_time, v_loss, v_accuracy, images_per_sec, total_time
        ))
        # Early stopping
        if len(val_accuracy) >= PATIENCE and all(acc >= TARGET_ACCURACY for acc in val_accuracy[-PATIENCE:]):
            print('Early stopping after epoch {}'.format(epoch + 1))
            break
    # Final metrics
    print("\nTraining complete. Final Validation Accuracy = {:5.3f}".format(val_accuracy[-1]))
    print("Total Training Time: {:5.3f} seconds".format(total_time))
def main():
    # Set the compute device
    device = get_device()
    # Train the WideResNet model
    train_resnet_without_ddp(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
if __name__ == "__main__":
    main()
````

This file contains the data utility functions used for preparing and managing the dataset.Please note that, you need to manually install the Fashion-MNIST dataset and place it in the respective folder.
````python
# dataset_utils.py
# NUmpy is a fundamental package for scientific computing. It contains an implementation of an array
import numpy as np
# to generate our own dataset
import random 
import torchvision
import torchvision.transforms as transforms
import torch



def load_fashion_mnist(batch_size, train_subset_size=10000, test_subset_size=10000):
    """
    Loads and preprocesses the Fashion-MNIST dataset.
    Args:
        batch_size (int): Batch size for training and testing.
        train_subset_size (int): Number of training samples to use (default:10,000) which we used initially for testing purpose.
        test_subset_size (int): Number of testing samples to use (default: 10,000) which we used initially for testing purpose.
    Returns:
        train_loader, test_loader: Data loaders for training and testing.
    """
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # Load full Datasets
    # Note that, if there is no internet access on the compute node, we need to manually download the 
    # datasets and then use it, for which we need to set download=False
    train_set = torchvision.datasets.FashionMNIST("/cluster/work/users/<user_name>/deepLearning/private/shared/data", download=False, transform=transform)
    test_set = torchvision.datasets.FashionMNIST("/cluster/work/users/<user_name>/deepLearning/private/shared/data", download=False, train=False, transform=transform)
    # Create subsets
    train_subset= torch.utils.data.Subset(train_set, list(range(0, train_subset_size)))
    test_subset= torch.utils.data.Subset(test_set, list(range(0, test_subset_size)))
    # Create the data loaders
    train_loader = torch.utils.data.Dataloader(train_subset, batch_size=batch_size, drop_last=True)
    test_loader = torch.utils.data.Dataloader(test_subset, batch_size, drop_last=True)
    return train_loader, test_loader


def load_fashion_mnist_fulldataset(batch_size):
    #Define transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # Load full datasets
    train_set = torchvision.datasets.FashionMNIST("/cluster/work/users/<user_name>/deepLearning/private/shared/data", download=False, transform=transform)
    test_set = torchvision.datasets.FashionMNIST("/cluster/work/users/<user_name>/deepLearning/private/shared/data", download=False, train=False, transform=transform)
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, dropLast=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, dropLast=True, shuffle=False)
    return train_loader, test_loader
````

This file includes the device utility functions, which handle device selection and management for training (e.g., selecting the appropriate GPU). 
````python
# device_utils.py
import torch

def get_device():
    """
    Determine the compute device (GPU or CPU).
    Returns:
        torch.device: The device to use for the computations.
    """

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
````


This file contains the implementation of the ResNet model architecture.
````python
# wide_resnet.py
import torch.nn as nn

# Standard convulation block followed by batch normalization
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1), padding='same', bias=False),nn.BatchNorm2d(output_channels), nn.ReLU())
    def forward(self, x):
        return self.cbr(x)
    
# Basic residual block
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels,output_channels, kernel_size=1, stride=(1,1), padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
    def forward(self,x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        return out + residual

# WideResnet model
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [1, 16, 160, 320, 640]
        self.input_block = cbrblock(nChannels[0], nChannels[1])
        self.block1 = conv_block(nChannels[1], nChannels[2], scale_input=True)
        self.block2 = conv_block(nChannels[2], nChannels[2], scale_input=False)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], scale_input=True)
        self.block4 = conv_block(nChannels[3], nChannels[3], scale_input=False)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], scale_input=True)
        self.block6 = conv_block(nChannels[4], nChannels[4], scale_input=False)
        # Global Average pooling
        self.pool = nn.AvgPool2d(7)
        # Fully connected layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

````

Finally, this file serves as a utility module for importing the training and testing datasets.
````python
# train_utils.py
import torch 
def train(model, optimizer, train_loader, loss_fn, device):
    """
    Trains the model for one epoch.
    Args:
        model(torch.nn.Module): The model to train.
        optimizer(torch.optim.Optimizer): Optimizer for updating model parameters.
        train_loader(torch.utils.data.DataLoader): DataLoader for training data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run training on (CPU or GPU).
    """
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward passs
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def test(model, test_loader, loss_fn, device):
    """
    Evaluates the model on the validation dataset.
    Args:
        model(torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Device to run evaluation on (CPU or GPU).
    Returns:
        tuple: Validation accuracy and validaiton loss.
    """

    model.eval()
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Compute accuracy and loss
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum().item()
            loss_total += loss.item()

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)
    return v_accuracy, v_loss
````


#### Job Script for Single GPU Training

To run the training on a single GPU, we use the following job script. The `accel` partition is used to access GPU resources. After loading the Python module from `cray-python`, you can create a virtual environment and install all the required wheels for your program using `pip`.

For resource management, we use the `torchrun` utility from PyTorch. This tool ensures efficient allocation of resources, especially when scaling to multiple GPUs or nodes.
````bash
#!/bin/bash
#SBATCH --job-name=simple_nn_training
#SBATCH --account=<project_number>
#SBATCH --output=singlenode.out
#SBATCH --error=singlenode.err
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1                     # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=72           # Reserve 72 CPU cores
#SBATCH --mem-per-gpu=110G           # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=1            # Request 1 GPU
# Load required modules
module load cray-python/3.11.7

# Create and activate virtual environment in compute node's local storage
VENV_PATH="$SCRATCH/pytorch_venv"  # Using compute node's scratch space
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# Install PyTorch from the wheel (offline installation)
WHEEL_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/torch_wheels"
pip install --no-index --find-links=$WHEEL_DIR $(ls $WHEEL_DIR/*.whl | tr '\n' ' ')


# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared:$PYTHONPATH

# Run the Python script using torchrun 

torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE ../resnet.py

deactivate
````

Output of the training is shown below:

````bash
Training WideResNet on Fashion MNIST with Batch Size: 32
Epoch =  1: Epoch Time = 2.963, Validation Loss = 0.524, Validation Accuracy = 0.799, Images/sec = 3369.181, Cumulative Time = 2.963
Epoch =  2: Epoch Time = 2.623, Validation Loss = 0.440, Validation Accuracy = 0.837, Images/sec = 3806.211, Cumulative Time = 5.586
Epoch =  3: Epoch Time = 2.601, Validation Loss = 0.422, Validation Accuracy = 0.849, Images/sec = 3839.022, Cumulative Time = 8.187
Epoch =  4: Epoch Time = 2.553, Validation Loss = 0.393, Validation Accuracy = 0.862, Images/sec = 3909.946, Cumulative Time = 10.741
Epoch =  5: Epoch Time = 2.553, Validation Loss = 0.424, Validation Accuracy = 0.859, Images/sec = 3911.137, Cumulative Time = 13.293
Early stopping after epoch 5

Training complete. Final Validation Accuracy = 0.859
Total Training Time: 13.293 seconds
````
### Multi-GPU Implementation
To scale our training to multiple GPUs, we will utilize PyTorch's Distributed Data Parallel (DDP) framework. DDP allows us to efficiently scale training across multiple GPUs and even across multiple nodes.

For this, we need to modify the main Python script to include DDP implementation. The updated script will work for both scenarios Multiple GPUs within a single node and Multiple nodes.

````python
# resnetddp.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from data.dataset_utils import load_fashion_mnist_fulldataset
from models.wide_resnet import WideResNet
from training.train_utils import train, test
# Parse input arguments
parser = argparse.ArgumentParser(description='Fashion MNIST DDP example',
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
def prepare_dataloader(dataset, batch_size):
    """Prepare DataLoader with DistributedSampler."""
    sampler = DistributedSampler(dataset, drop_last=False)  # Ensure no data is dropped
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader, sampler
def main_worker():
    ddp_setup()
    # Get the local rank and device
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")


    # Log initialization info
    if global_rank == 0:
        print(f"Training started with {world_size} processes across {world_size // torch.cuda.device_count()} nodes.")
        print(f"Using {torch.cuda.device_count()} GPUs per node.")

    
    # Load the dataset
    train_loader, test_loader = load_fashion_mnist_fulldataset(batch_size=args.batch_size)
    # Create the model and wrap it with DDP
    num_classes = 10
    model = WideResNet(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank])
   
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)
    val_accuracy = []
    total_time = 0
    # Training loop
    for epoch in range(args.epochs):
        if global_rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        # Train the model for one epoch
        t0 = time.time()
        train(model, optimizer, train_loader, loss_fn, device)
        # Synchronize all processes
        torch.distributed.barrier()
        epoch_time = time.time() - t0
        total_time += epoch_time
        # Compute validation accuracy and loss
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        # Average validation metrics across all GPUs
        v_accuracy_tensor = torch.tensor(v_accuracy).to(device)
        v_loss_tensor = torch.tensor(v_loss).to(device)
        torch.distributed.all_reduce(v_accuracy_tensor, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(v_loss_tensor, op=torch.distributed.ReduceOp.AVG)
        # Print metrics only from the main process
        if global_rank == 0:
            print(f"Epoch {epoch + 1} completed in {epoch_time:.3f} seconds")
            print(f"Validation Loss: {v_loss_tensor.item():.4f}, Validation Accuracy: {v_accuracy_tensor.item():.4f}")
        # Early stopping
        val_accuracy.append(v_accuracy_tensor.item())
        if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
            if global_rank == 0:
                print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break
    
       # Log total training time and summary
    if global_rank == 0:
        print("\nTraining Summary:")
        print(f"Total training time: {total_time:.3f} seconds")
        print(f"Number of nodes: {world_size // torch.cuda.device_count()}")
        print(f"Number of GPUs per node: {torch.cuda.device_count()}")
        print(f"Total GPUs used: {world_size}")
        print("Training completed successfully.")
    # Clean up the distributed environment
    destroy_process_group()
if __name__ == '__main__':
    main_worker()
````

#### Job Script for Multi GPU Training

To run the training on multiple GPUs, we can use the same job script mentioned earlier, but specify a higher number of GPUs.

When using `torchrun` for a single-node setup, you need to include the `standalone` argument. However, this argument is not required for a multi-node setup. The full job script is given below:

````bash
#!/bin/bash
#SBATCH --job-name=resnet_training_singlenode
#SBATCH --account=<project_number>
#SBATCH --output=singlenode.out
#SBATCH --error=singlenode.err
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1     # Use one compute node
#SBATCH --ntasks-per-node=1 #  Single task per node
#SBATCH --cpus-per-task=72  # Reserve enough CPU cores for full workload
#SBATCH --mem-per-gpu=110G   # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=2   # Reserve 2 GPUs on node
# Load required modules
module load cray-python/3.11.7

# Create and activate virtual environment in compute node's local storage
VENV_PATH="$SCRATCH/pytorch_venv"  # Using compute node's scratch space
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# Install PyTorch from the wheel (offline installation)
WHEEL_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/torch_wheels"
pip install --no-index --find-links=$WHEEL_DIR $(ls $WHEEL_DIR/*.whl | tr '\n' ' ')


# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared:$PYTHONPATH

# Run the Python script using torchrun 

torchrun --standalone --nnodes=$SLURM_NNODES  --nproc_per_node=$SLURM_GPUS_ON_NODE ../resnetddp.py  --epochs 10 --batch-size 512

deactivate
````
Output of the training is shown below:

````bash
Training started with 2 processes across 1 nodes.
Using 2 GPUs per node.

Epoch 1/10
Epoch 1 completed in 7.350 seconds
Validation Loss: 0.5492, Validation Accuracy: 0.7971

Epoch 2/10
Epoch 2 completed in 7.150 seconds
Validation Loss: 0.4315, Validation Accuracy: 0.8414

Epoch 3/10
Epoch 3 completed in 7.090 seconds
Validation Loss: 0.3536, Validation Accuracy: 0.8744

Epoch 4/10
Epoch 4 completed in 7.112 seconds
Validation Loss: 0.3402, Validation Accuracy: 0.8759
Target accuracy reached. Early stopping after epoch 4.

Training Summary:
Total training time: 28.702 seconds
Number of nodes: 1
Number of GPUs per node: 2
Total GPUs used: 2
Training completed successfully.
````


### Multi-Node Setup

Setting up training across multiple nodes is relatively straightforward since we use the same Python script as in the multi-GPU implementation. The main difference lies in using a different job script, which is provided below.

For multi-node jobs, a few key considerations are important:

1. Communication Interface: You need to specify the communication interface to enable proper communication between nodes.

2. Master Node: The master node must be designated to handle coordination and communication across nodes.

We use srun to launch the job across multiple nodes, allowing torchrun to efficiently manage and coordinate the training process.

#### Job Script for Multi node Training

````bash
#!/bin/bash
#SBATCH --job-name=resnet_training_multinode
#SBATCH --account=<project_number>
#SBATCH --output=multinode.out
#SBATCH --error=multinode.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=2                 # Request 2 compute nodes
#SBATCH --ntasks-per-node=1       # One coordinating task per node
#SBATCH --gpus-per-node=4        # Reserve 4 GPUs on each node
#SBATCH --cpus-per-task=72       # Reserve enough CPU cores for full workload
#SBATCH --mem-per-gpu=110G        # Request 110 GB of CPU RAM per GPU

# Load required modules
module load cray-python/3.11.7

# Create and activate virtual environment
VENV_PATH="$SCRATCH/pytorch_venv"
python -m venv $VENV_PATH
source $VENV_PATH/bin/activate
# Install PyTorch from the wheel (offline installation)
WHEEL_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/torch_wheels"
pip install --no-index --find-links=$WHEEL_DIR $(ls $WHEEL_DIR/*.whl | tr '\n' ' ')

# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared:$PYTHONPATH

# Set NCCL environment variables for debugging and communication
# export NCCL_DEBUG=INFO  # Use it to see details
export NCCL_SOCKET_IFNAME=hsn0  # Replace with hsn1 if needed

# Get the head node and its IP address
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"

# Run the Python script using torchrun 
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  ../resnetddp.py
# Deactivate the virtual environment
deactivate
````

Below is the output generated from running the training across multiple nodes.

````bash

Training started with 4 processes across 2 nodes.
Using 2 GPUs per node.

Epoch 1/10
Epoch 1 completed in 10.208 seconds
Validation Loss: 0.5929, Validation Accuracy: 0.7752

Epoch 2/10
Epoch 2 completed in 9.450 seconds
Validation Loss: 0.4266, Validation Accuracy: 0.8477

Epoch 3/10
Epoch 3 completed in 9.385 seconds
Validation Loss: 0.4440, Validation Accuracy: 0.8397

Epoch 4/10
Epoch 4 completed in 9.400 seconds
Validation Loss: 0.4783, Validation Accuracy: 0.8431

Epoch 5/10
Epoch 5 completed in 9.436 seconds
Validation Loss: 0.3952, Validation Accuracy: 0.8575

Epoch 6/10
Epoch 6 completed in 9.526 seconds
Validation Loss: 0.2980, Validation Accuracy: 0.8931
Target accuracy reached. Early stopping after epoch 6.

Training Summary:
Total training time: 57.405 seconds
Number of nodes: 2
Number of GPUs per node: 2
Total GPUs used: 4
Training completed successfully.
````

