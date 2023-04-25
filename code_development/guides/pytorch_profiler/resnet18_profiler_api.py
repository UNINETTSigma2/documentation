#import all the necessary libraries 
import torch
import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights

#prepare input data and transform it
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), 
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,transform=transform)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
    download=True, transform=transform)

# use dataloader to launch each batch
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=1)

# Create a Resnet model, loss function, and optimizer objects. To run on GPU, move model and loss to a GPU device
device = torch.device("cuda:0")

#model = torchvision.models.resnet18(pretrained=True).cuda(device)
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).cuda(device)
criterion = torch.nn.CrossEntropyLoss().cuda(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.train()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result-4batch', worker_name='worker4'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as prof:
#include
    for step, data in enumerate(trainloader):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step >= 10:
            break
        prof.step()

print()
print(f'--Print GPU: {torch.cuda.device_count()}')
print(torch.cuda.is_available())
