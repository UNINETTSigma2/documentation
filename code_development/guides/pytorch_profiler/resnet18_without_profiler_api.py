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

#Define the training step for each batch of input data.
def train(data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print()
print(f'--Print GPU: {torch.cuda.device_count()}')
print(torch.cuda.is_available())
