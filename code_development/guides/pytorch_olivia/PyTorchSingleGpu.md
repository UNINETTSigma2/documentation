(pytorch-on-olivia)=
# PyTorch on Olivia

```{contents}
:depth: 2
```

This guide demonstrates how to run PyTorch on Olivia using NVIDIA's optimized [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). We train a Wide ResNet model on the CIFAR-100 dataset across three scenarios:

1. **Single GPU** (this page)
2. **Multi-GPU** - 4 GPUs on a single node ({ref}`pytorch-multi-gpu`)
3. **Multi-Node** - Multiple nodes ({ref}`pytorch-multi-node`)

```{admonition} Performance Tips
:class: tip

**Single-GPU optimization:** For maximum single-GPU performance (~1.6x speedup), use 32 workers, BFloat16 precision, and prefetching. However, these optimizations should *not* be used for multi-GPU training.

**Multi-GPU/Multi-node:** The baseline configurations in this guide (8 workers, FP16) are already optimal. Using more workers actually degrades DDP performance due to gradient synchronization pacing—the data arrives faster than gradients can sync between GPUs.
```

```{note}
**Key considerations for Olivia:**
- The login node (x86_64) and compute nodes (Aarch64) have different architectures. Software must be run via containers on the compute nodes.
- Compute nodes use CUDA 12.7. Ensure container compatibility.
```

## Getting the Container

The PyTorch container is available pre-pulled at:
```
/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif
```

To pull a different version yourself, use the `--arch arm64` flag since you're pulling from the login node (x86_64) for use on compute nodes (Aarch64):

```bash
apptainer pull --arch arm64 docker://nvcr.io/nvidia/pytorch:25.06-py3
```

## Project Setup

```{warning}
Set up your project in your **work or project area** (e.g., `/cluster/work/projects/nnXXXXk/`), not your home directory. The CIFAR-100 dataset (~500 MB) will be downloaded automatically on first run.
```

Create a directory with all files in a flat structure:
```
/cluster/work/projects/nnXXXXk/username/pytorch_test/
├── train.py
├── train_ddp.py          # for multi-GPU/multi-node
├── dataset_utils.py
├── train_utils.py
├── device_utils.py
├── model.py
├── singlegpu_job.sh
├── multigpu_job.sh       # for multi-GPU
├── multinode_job.sh      # for multi-node
└── datasets/             # created automatically on first run
    └── cifar-100-python/
```

## Single GPU Implementation

To train the Wide ResNet model on a single GPU, we used the following files. The `train.py` file include the main Python script used for training the Wide ResNet model.
```{code-block} python
:linenos:

#train.py

"""
This script trains a Wide ResNet model on the CIFAR-100  dataset that is a  single-GPU implementation of the training process.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Import custom modules
from dataset_utils import load_cifar100
from model  import WideResNet
from train_utils import train as train_one_epoch, test as evaluate
from device_utils import get_device


# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.01
TARGET_ACCURACY = 0.95
PATIENCE = 2

def run_train(batch_size, epochs, learning_rate, device):
    """
    Trains a WideResNet model on the CIFAR-100 dataset for single GPU.
    Args:
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run training on (CPU or GPU).
    Returns:
        throughput (float): Images processed per second
    """

    print(f"Training WideResNet on CIFAR-100  with Batch Size: {batch_size}")

    # Training variables
    val_accuracy = []
    total_time = 0
    total_images = 0 # total images processed

    # Load the dataset
    train_loader, test_loader = load_cifar100(batch_size=batch_size)

    # Initialize the WideResNet Model
    num_classes = 100    # num_class set to  100 for CIFAR-100.
    model = WideResNet(num_classes).to(device)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        t0 = time.time()

        # Train the model for one epoch
        train_one_epoch(model, optimizer, train_loader, loss_fn, device)

        # Calculate epoch time
        epoch_time = time.time() - t0
        total_time += epoch_time

        # Compute throughput (images per second)
        images_per_sec = len(train_loader) * batch_size / epoch_time
        total_images += len(train_loader) * batch_size

        # Compute validation accuracy and loss
        v_accuracy, v_loss = evaluate(model, test_loader, loss_fn, device)
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

    # Train the WideResNet model to get the throughput(i.e. images processed per second)
    throughput = run_train(batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, device=device)
    print(f"Single-GPU Thrpughput: {throughput:.3f} images/second")

if __name__ == "__main__":
    main()
```

The `dataset_utils.py` file contains the data utility functions used for preparing and managing the dataset.Please note that, you will be installing the CIFAR-100 dataset and it will be placed in the `datasets` folder through this script.

```{code-block} python
:linenos:

# dataset_utils.py

import torchvision
import torchvision.transforms as transforms
import torch
from pathlib import Path
import os


def _data_dir_default():
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir




def load_cifar100(batch_size, num_workers=0,sampler=None, data_dir=None):
    """
    Loads the CIFAR-100 dataset.Create the dataset directory to store dataset during runtime and no environment variable support.
    """
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()

    # Define transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 mean and std
    ])

    # Load full datasets
    train_set = torchvision.datasets.CIFAR100(
            root=str(root),download=True,train=True,transform=transform)
    test_set = torchvision.datasets.CIFAR100(root=str(root),download=True,train=False,transform=transform)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,drop_last=True,shuffle=(sampler is None),sampler= sampler,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,drop_last=True,shuffle=False,num_workers=num_workers,pin_memory=True)
    return train_loader, test_loader
```

The `device_utils.py` file includes the device utility functions, which handle device selection and management for training (e.g., selecting the appropriate GPU).

```{code-block} python
:linenos:

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

The `model.py` file contains the implementation of the Wide ResNet model architecture.

```{code-block} python
:linenos:

# model.py

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

Finally, the `train_utils.py` file serves as a utility module for importing the training and testing datasets.

```{code-block} python
:linenos:

# train_utils.py
import torch

def train(model, optimizer, train_loader, loss_fn, device):
    """
    Trains the model for one epoch.Note that, this function will be used only for single gpu implementation. For the multi-gpu implementation, we will be defining the train function in the train_ddp.py file itself.
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
    Evaluates the model on the validation dataset.Note that, this function will be used in the multi-gpu implementation aswell.
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

## Job Script for Single GPU Training

The `--nv` flag gives the container access to GPU resources. We use `torchrun` to launch the training script.

```{code-block} bash
:linenos:

#!/bin/bash
#SBATCH --job-name=resnet_singleGpu
#SBATCH --account=<project_number>
#SBATCH --output=singlegpu_%j.out
#SBATCH --error=singlegpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --gpus-per-node=1

CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.06_arm64.sif"

cd "${SLURM_SUBMIT_DIR}"

# Check GPU availability
apptainer exec --nv $CONTAINER_PATH python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")'

# Run training
apptainer exec --nv $CONTAINER_PATH torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py
```

Example output showing training progress:


```bash
Epoch = 95: Epoch Time = 19.061, Validation Loss = 1.352, Validation Accuracy = 0.730, Images/sec = 2622.313, Cumulative Time = 1828.359
Epoch = 96: Epoch Time = 19.087, Validation Loss = 1.339, Validation Accuracy = 0.735, Images/sec = 2618.747, Cumulative Time = 1847.446
Epoch = 97: Epoch Time = 19.006, Validation Loss = 1.306, Validation Accuracy = 0.741, Images/sec = 2629.863, Cumulative Time = 1866.452
Epoch = 98: Epoch Time = 19.072, Validation Loss = 1.308, Validation Accuracy = 0.739, Images/sec = 2620.757, Cumulative Time = 1885.524
Epoch = 99: Epoch Time = 19.056, Validation Loss = 1.317, Validation Accuracy = 0.736, Images/sec = 2623.024, Cumulative Time = 1904.580
Epoch = 100: Epoch Time = 19.074, Validation Loss = 1.316, Validation Accuracy = 0.740, Images/sec = 2620.481, Cumulative Time = 1923.655

Training complete. Final Validation Accuracy = 0.740
Total Training Time: 1923.655 seconds
Throughput: 2598.388 images/second
Single-GPU Thrpughput: 2598.388 images/second
```
The output suggests that the total throughput that we obtained from single GPU training is ` 2598.388 images/second` and it took approximately `1923.655 seconds` to complete the training. As we proceed forward with the multi-gpu implementation, our goal would be to achieve higher throughput and also possibly reduced the training time for the same number of epochs.


Now the goal is to scale this up to multiple GPUs.For this, please check out the {ref}`Multi GPU Guide <pytorch-multi-gpu>`.

```{admonition} Exercise
:class: tip

Try changing the number of workers in `dataset_utils.py` from 0 to 32 and observe the performance difference. With 32 workers on the GH200's Grace CPU, you should see approximately 1.6x speedup (~4,200 images/second vs ~2,600 images/second).
```
