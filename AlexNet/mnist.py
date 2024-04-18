import torch, torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils
from alexnet import AlexNet

transform = transforms.Compose([
    transforms.Resize((227, 227)), # the original mnist image size is 28*28, alexnet input size is 227*227
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

train = MNIST(root='./data', train=True, download=True, transform=transform)
test = MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

model = AlexNet(10, 1)
