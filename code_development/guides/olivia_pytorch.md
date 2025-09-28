(pytorch-wheels)=

# PyTorch on Olivia

```{contents}
:depth: 3
```

In this guide, we’ll be testing PyTorch on the Olivia system, which uses the Aarch64 architecture on its compute nodes. To do this, we’ll use  PyTorch container from Nvidia.The process of training a ResNet model using a containerized environment, will bypass the need to manually download and install PyTorch wheels. The container includes all the necessary packages required to run the project, simplifying the setup process.

## Key Considerations

1. __Different Architectures__:

     The login node and the compute node on Olivia have different architectures. The login node uses the x86_64 architecture, while the compute node uses Aarch64. This means we cannot install software directly on the login node and expect it to work on the compute node.


2. CUDA Version:

     The compute nodes are equipped with CUDA Version 12.7, as confirmed by running the nvidia-smi command. Therefore, we need to ensure that the container we will be using will be compatible with this CUDA version.


## Training a ResNet Model with the CIFAR-100 Dataset   

To test Olivia's capabilities with real-world workloads, we will train a ResNet model using the CIFAR-100 dataset. The testing will be conducted under the following scenarios:

1. Single GPU

2. Multiple GPUs

3. Multiple Nodes

The primary goal of this exercise is to verify that we can successfully run training tasks on Olivia. As such, we will not delve into the specifics of neural network training in this documentation. A separate guide will be prepared to cover those details.

### Single GPU Implementation

To train the ResNet model on a single GPU, we used the following files. These files include the main Python script responsible for training the ResNet model.

```python
# resnet.py
"""
This script trains a WideResNet model on the CIFAR-100  dataset without using Distributed Data Parallel (DDP).
The goal is to provide a simple, single-GPU implementation of the training process.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys
# Import custom modules
from data.dataset_utils import load_fashion_mnist_fulldataset, load_cifar100
from models.wide_resnet import WideResNet
from training.train_utils import train, test
from utils.device_utils import get_device

# Define paths
shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../shared"))
images_dir = os.path.join(shared_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
TARGET_ACCURACY = 0.95
PATIENCE = 2

def train_resnet_without_ddp(batch_size, epochs, learning_rate, device):
    """
    Trains a WideResNet model on the CIFAR-100 dataset without DDP.
    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
    Returns:
        throughput (float): Images processed per second
    """
    print(f"Training WideResNet on CIFAR 100 with Batch Size: {batch_size}")
    # Training variables
    val_accuracy = []
    total_time = 0
    total_images = 0 # total images processed
    # Load the dataset
    train_loader, test_loader = load_cifar100(batch_size=batch_size)
    # Initialize the WideResNet Model
    # For fashion MNIST dataset
    #num_classes = 10

    # For CIFAR-100 dataset
    num_classes = 100
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
        total_images += len(train_loader) * batch_size
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
    throughput = total_images / total_time
    print("\nTraining complete. Final Validation Accuracy = {:5.3f}".format(val_accuracy[-1]))
    print("Total Training Time: {:5.3f} seconds".format(total_time))
    print("Throughput: {:5.3f} images/second".format(throughput))
    return throughput
def main():
    # Set the compute device
    device = get_device()
    # Train the WideResNet model
    throughput = train_resnet_without_ddp(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
    print(f"Single-GPU Thrpughput: {throughput:.3f} images/second")
if __name__ == "__main__":
    main()
```


This file contains the data utility functions used for preparing and managing the dataset.Please note that, you will be installing the CIFAR-100 dataset and it will be placed in the respective folder through the job script.

```python
# dataset_utils.py
import torchvision
import torchvision.transforms as transforms
import torch

def load_cifar100(batch_size, num_workers=0, sampler=None):
    """
    Loads the CIFAR-100 dataset.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 mean and std
    ])
    # Load full datasets
    train_set = torchvision.datasets.CIFAR100(
            "/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared/data/",download=True,train=True,transform=transform)
    test_set = torchvision.datasets.CIFAR100("/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared/data/",download=True,train=False,transform=transform)
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,drop_last=True,shuffle=(sampler is None),sampler= sampler ,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,drop_last=True,shuffle=False,num_workers=num_workers,pin_memory=True)
    return train_loader, test_loader
```

This file includes the device utility functions, which handle device selection and management for training (e.g., selecting the appropriate GPU). 
```python
# device_utils.py
import torch

def get_device():
    """
    Determine the compute device (GPU or CPU).
    Returns:
        torch.device: The device to use for the computations.
    """

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


This file contains the implementation of the ResNet model architecture.
```python
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
        # RGB images (3 channels) input for CIFAR-100 dataset
        nChannels = [3, 16, 160, 320, 640]
        # Grayscale images (1 channel) for Fashion MNIST dataset
        # nChannels = [1, 16, 160, 320, 640]
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

```

Finally, this file serves as a utility module for importing the training and testing datasets.
```python
# train_utils.py
import torch
def train(model, optimizer, train_loader, loss_fn, device):
    """
    Trains the model for one epoch. Note, that we will use this only for single GPU implmentation.
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
```


Since we are using the container, to provide the container access to additional directories, such as the one containing the training scripts, we must explicitly bind these directories using the `--bind` option.

The job scripts below shows how it is done for accessing the required files inside the container.

Note that, the command to run the script includes the `--nv` option, which ensures that the container has access to GPU resources. This is essential for leveraging hardware acceleration during training.

#### Job Script for Single GPU Training

In the job script below, we will be using a single GPU for training.The container we will be using is downloaded and placed in this path `/cluster/projects/nn9999k/jorn/pytorch_25.05-py3.sif`. This container contains all the necessary packages for running our deep learning training. for instance. torch and torchvision.Moreover, we are binding the directory from the host inside the container using this `BIND_DIR="/cluster/work/projects/nn9997k/binod/PyTorch/private"`. Also, to include the shared directory to import other python files as the modules, we exported the python path as shown here `BIND_DIR="/cluster/work/projects/nn9997k/binod/PyTorch/private"`. Finally, we run the `apptainer exec --nv` command binding the directory from the host and using torchrun.

```bash
#!/bin/bash
#SBATCH --job-name=simple_nn_training
#SBATCH --account=<project_number>
#SBATCH --time=00:10:00
#SBATCH --output=resnet_with_container_%j.out
#SBATCH --error=resnet_with_container_%j.err
#SBATCH --partition=accel
#SBATCH --nodes=1                     # Single compute node
#SBATCH --nodelist=x1000c2s4b1n0
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=72           # Reserve 72 CPU cores
#SBATCH --mem-per-gpu=110G           # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=1            # Request 1 GPU

# Path to the container
CONTAINER_PATH="/cluster/projects/nn9999k/jorn/pytorch_25.05-py3.sif"

# Path to the training script
TRAINING_SCRIPT="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/simple_nn_project/resnet.py"

# Bind the directory
BIND_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private"


# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/nn9997k/binod/PyTorch/private/shared:$PYTHONPATH

# Check GPU availability inside the container
echo "Checking GPU availability inside the container..."
apptainer exec --nv --bind $BIND_DIR $CONTAINER_PATH python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="gpu_utilization_resnet_with_container.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &l

# Run the training script with torchrun inside the container
apptainer exec --nv --bind $BIND_DIR $CONTAINER_PATH \ torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT


# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"

```


Output of the training is shown below:

```bash
Epoch = 95: Epoch Time = 20.235, Validation Loss = 1.331, Validation Accuracy = 0.739, Images/sec = 2470.132, Cumulative Time = 1893.528
Epoch = 96: Epoch Time = 20.331, Validation Loss = 1.313, Validation Accuracy = 0.743, Images/sec = 2458.565, Cumulative Time = 1913.859
Epoch = 97: Epoch Time = 20.173, Validation Loss = 1.325, Validation Accuracy = 0.742, Images/sec = 2477.718, Cumulative Time = 1934.032
Epoch = 98: Epoch Time = 20.168, Validation Loss = 1.328, Validation Accuracy = 0.736, Images/sec = 2478.441, Cumulative Time = 1954.200
Epoch = 99: Epoch Time = 20.065, Validation Loss = 1.323, Validation Accuracy = 0.739, Images/sec = 2491.122, Cumulative Time = 1974.265
Epoch = 100: Epoch Time = 19.773, Validation Loss = 1.332, Validation Accuracy = 0.739, Images/sec = 2527.948, Cumulative Time = 1994.037

Training complete. Final Validation Accuracy = 0.739
Total Training Time: 1994.037 seconds
Throughput: 2506.673 images/second
Single-GPU Thrpughput: 2506.673 images/second
```
The output suggests that the total throughput that we obtained from single GPU training is ` 2506.673 images/second` and it took approximately `1994.037 seconds` to complete the training. As we proceed forward with the multi-gpu implementation, our goal would be to achieve higher throughtput and also possibly reduced the training time.


### Multi-GPU Implementation
To scale our training to multiple GPUs, we will utilize PyTorch's Distributed Data Parallel (DDP) framework. DDP allows us to efficiently scale training across multiple GPUs and even across multiple nodes.To learn more about how to implement DDP, please refer to this official documentation from PyTorch [Getting Started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)

For this, we need to modify the main Python script to include DDP implementation. The updated script will work for both scenarios Multiple GPUs within a single node and Multiple nodes.

```python
# singlenode.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from data.dataset_utils import load_cifar100
from models.wide_resnet import WideResNet
from training.train_utils import test 
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
    device = torch.device(f"cuda:{local_rank}")
    # Log initialization info
    if global_rank == 0:
        print(f"Training started with {world_size} processes across {world_size // torch.cuda.device_count()} nodes.")
        print(f"Using {torch.cuda.device_count()} GPUs per node.")
    # Load the CIFAR-100 dataset with DistributedSampler
    per_gpu_batch_size = args.batch_size // world_size  # Divide global batch size across GPUs
    train_sampler = DistributedSampler(
        torchvision.datasets.CIFAR100(
            root="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared/data/",
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

#### Job Script for Multi GPU Training

To run the training on multiple GPUs, we can use the same job script mentioned earlier, but specify a higher number of GPUs.

When using `torchrun` for a single-node setup, you need to include the `--standalone` argument. However, this argument is not required for a multi-node setup. 
This job script is designed to train the model across multiple GPUs on a single node. In this script, we explicitly define the path to the torchrun executable using the following line:

`TORCHRUN_PATH="/usr/local/bin/torchrun"`

While experimenting, we encountered cases where the torchrun executable was not recognized unless its full path was explicitly specified. Defining the TORCHRUN_PATH in the script resolves this issue. However, this configuration may vary depending on your working environment:

If the torchrun executable is already in your $PATH, explicitly setting the path may not be necessary.



```bash
#!/bin/bash
#SBATCH --job-name=simple_nn_training
#SBATCH --account=<project_number>
#SBATCH --output=singlenode_with_container_%j.out
#SBATCH --error=singlenode_with_container_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1     # Use one compute node
##SBATCH --nodelist=x1000c2s4b1n0
#SBATCH --ntasks-per-node=1 #  Single task per node
#SBATCH --cpus-per-gpu=72  # Reserve enough CPU cores for full workload with each GPU
#SBATCH --mem-per-gpu=110G   # Request 110 GB of CPU RAM per GPU
#SBATCH --gpus-per-node=4   # Reserve 4 GPUs on node

# Path to the container
CONTAINER_PATH="/cluster/projects/nn9999k/jorn/pytorch_25.05-py3.sif"

# Path to the training script
TRAINING_SCRIPT="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/simple_nn_project/singlenode.py --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2"

# Bind directories
BIND_DIR="/cluster/work/projects/<project_number>/<user_name>/PyTorch/private"

# Set PYTHONPATH to include the shared directory
export PYTHONPATH=/cluster/work/projects/<project_number>/<user_name>/PyTorch/private/shared:$PYTHONPATH

# Explicitly specify the full path to torchrun
TORCHRUN_PATH="/usr/local/bin/torchrun"

apptainer exec --nv $CONTAINER_PATH which torchrun
# Start GPU utilization monitoring in the background
GPU_LOG_FILE="gpu_utilization_multinode_container.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

# Run the training script with torchrun inside the container
apptainer exec --nv --bind $BIND_DIR $CONTAINER_PATH $TORCHRUN_PATH --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"

```

Output of the training is shown below:

```bash
Epoch 95/100 completed in 1.280 seconds
Validation Loss: 1.0222, Validation Accuracy: 0.7416
Epoch Throughput: 38409.065 images/second
Epoch 96/100 completed in 1.271 seconds
Validation Loss: 1.0204, Validation Accuracy: 0.7439
Epoch Throughput: 38665.960 images/second
Epoch 97/100 completed in 1.268 seconds
Validation Loss: 1.0401, Validation Accuracy: 0.7393
Epoch Throughput: 38766.180 images/second
Epoch 98/100 completed in 1.276 seconds
Validation Loss: 1.0070, Validation Accuracy: 0.7447
Epoch Throughput: 38512.740 images/second
Epoch 99/100 completed in 1.273 seconds
Validation Loss: 1.0075, Validation Accuracy: 0.7435
Epoch Throughput: 38609.904 images/second
Epoch 100/100 completed in 1.281 seconds
Validation Loss: 1.0194, Validation Accuracy: 0.7429
Epoch Throughput: 38355.399 images/second

Training Summary:
Total training time: 131.142 seconds
Throughput: 37479.949 images/second
Number of nodes: 1
Number of GPUs per node: 4
Total GPUs used: 4
Training completed successfully.
```

Note that, By using four GPUs now we can see that the throughput is  `37479.949 images/second` and training time is `131.142 seconds`.It suggests that we achieved superlinear scaling along with perfect scaling.The 4-GPU setup is highly efficient, with a speedup factor of `15.21` and a scaling efficiency of `374.1%`. This is the indication that  distributed training setup is highly optimized.

### Multi-Node Setup

We have successfully tested the multi-node setup using the native installation and verified that it scales effectively. However, to fully leverage Slingshot within the containerized environment, we discovered the need to install a custom version of NCCL integrated with the AWS-OFI plugin. This aspect is still under development, and we will update this documentation with the necessary details once the process is finalized.It might be that we will soon provide the container with necessary components installed in it.

In the meantime, if you are interested in learning how we achieved functionality with the native installation, please feel free to reach out to us. We will be happy to provide additional details and guidance.