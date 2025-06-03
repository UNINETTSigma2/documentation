#import all the necessary libraries 
import os
import torch
import torch.nn as nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main():
    ddp_setup()
    rank = int(os.environ["LOCAL_RANK"])
    
    # Prepare data
    transform = T.Compose([
        T.Resize(256), 
        T.CenterCrop(224), 
        T.ToTensor(), 
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model setup
    device = torch.device(f"cuda:{rank}")
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    
    prof = torch.profiler.profile(
      activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
      ],
      schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
      on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./outgpus',
        worker_name=f'worker{rank}'
      ),
      record_shapes=True,
      profile_memory=True,
      with_stack=True
    )
    prof.start()

    for step, data in enumerate(trainloader):
      inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
    
      outputs = model(inputs)
      loss = criterion(outputs, labels)
    
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
    
      prof.step()
      print(f"Rank {rank} - step: {step}, loss: {loss.item():.4f}")

    
      if step >= 10:
        break

    prof.stop()
    
    destroy_process_group()

if __name__ == "__main__":
    main()