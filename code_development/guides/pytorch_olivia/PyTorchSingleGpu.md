
(pytorch-on-olivia)=
# PyTorch on Olivia

```{contents}
:depth: 3
```

In this guide, we’ll be testing PyTorch on the Olivia system, which uses the Aarch64 architecture on its compute nodes. To do this, we’ll use  PyTorch container from Nvidia.The process of training a ResNet model using a containerized environment, will bypass the need to manually download and install PyTorch wheels. The container includes all the necessary packages required to run the project, simplifying the setup process. The container used in this tutorial is downloaded from here [PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.11-py3). This container is optimized to use with NVIDIA GPUs and contains the software such as CUDA, NVIDIA cuDNN , NVIDIA NCCL and so on.You can read more about it on the link given above. 

Note:  To replicate this project, create a directory within your project folder and place the following files inside it: `train.py`, `dataset_utils.py`, `train_utils.py`, `device_utils.py`, `model.py` inside it. 
Next, create a subdirectory called `scripts` within this directory and place the provided job script (shown below) inside it. This setup will be sufficient for quickly testing the single-GPU configuration.


## Key Considerations

1. __Different Architectures__:

     The login node and the compute node on Olivia have different architectures. The login node uses the x86_64 architecture, while the compute node uses Aarch64. This means we cannot install software directly on the login node and expect it to work on the compute node.


2. CUDA Version:

     The compute nodes are equipped with CUDA Version 12.7, as confirmed by running the nvidia-smi command. Therefore, we need to ensure that the container we will be using will be compatible with this CUDA version.


## Training a ResNet Model with the CIFAR-100 Dataset   

To test Olivia's capabilities with real-world workloads, we will train a Wide ResNet model using the CIFAR-100 dataset. The testing will be conducted under the following scenarios:

1. Single GPU

2. Multiple GPUs

3. Multiple Nodes

The primary goal of this exercise is to verify that we can successfully run training tasks on Olivia. As such, we will not delve into the specifics of neural network training in this documentation.

## Single GPU Implementation

To train the Wide ResNet model on a single GPU, we used the following files. The `train.py` file include the main Python script used for training the Wide ResNet model.

```python
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

```python
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


The `model.py` file contains the implementation of the Wide ResNet model architecture.

```python
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

```python

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


After placing these files in the working directory, you are ready to begin training the Neural Network. However, if you are replicating this experiment, we recommend creating a new directory within your working directory to store your job scripts. Organizing your files this way minimizes the need for modifications to the job scripts, allowing you to get them running quickly and efficiently.





#### Job Script for Single GPU Training

Note that, the command to run the script includes the `--nv` option, which ensures that the container has access to GPU resources. This is essential for leveraging hardware acceleration during training.

In the job script below, we will be using a single GPU for training.The container we will be using is downloaded and placed in this path `/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif`. If you have requirements for the newer version, please let us know so that we can assist you downloading and placing it on the same path.

Finally, we run the `apptainer exec --nv` using torchrun to run the training.


```bash
#!/bin/bash
#SBATCH --job-name=resnet_singleGpu
#SBATCH --account=<project_number>
#SBATCH --output=singlegpu_%j.out
#SBATCH --error=singlegpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=accel.           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=72           # Reserve 72 CPU cores(Eacho node has 256 CPUs)
#SBATCH --mem=110G                   # Request 110 GB RAM(Each node has 768 GiB total )
#SBATCH --gpus-per-node=1            # Request 1 GPU

# Path to the container
CONTAINER_PATH="/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif"

# Path to the training script
TRAINING_SCRIPT="train.py"

cd "${SLURM_SUBMIT_DIR}/.."

# Check GPU availability inside the container
echo "Checking GPU availability inside the container..."
apptainer exec --nv  $CONTAINER_PATH python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="singlegpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > $GPU_LOG_FILE &

# Run the training script with torchrun inside the container
apptainer exec --nv $CONTAINER_PATH torchrun --standalone --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE $TRAINING_SCRIPT

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
pkill -f "nvidia-smi --query-gpu"
```

Once you run this job script, you will be able see the output and error files inside the script directory and the gpu utilization could be verified from the `singlegpu.log` file outside script directory.Below is the output of the training which shows that the total training time and the throughput together with the Validation loss and Validation Accuracy.


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
The output suggests that the total throughput that we obtained from single GPU training is ` 2598.388 images/second` and it took approximately `1923.655 seconds` to complete the training. As we proceed forward with the multi-gpu implementation, our goal would be to achieve higher throughtput and also possibly reduced the training time for the same number of epochs.


Now the goal is to scale this up to multiple GPUs.For this, please check out the {ref}`Multi GPU Guide <pytorch-multi-gpu>`.



