import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np

from models import *
import torch.optim as optim

use_cuda=True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
'''some paremeters to figure out: '''



'-----------------------------------1. load data BEGIN-------------------------------------'
train_loader = data.DataLoader(
    datasets.CIFAR10(root='../data', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
    batch_size=4, shuffle=True)

'''show the image'''
# def imshow(img):
#     # img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     print(npimg.shape)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# dataiter = iter(train_loader)
# images, labels = next(dataiter)
# imshow(torchvision.utils.make_grid(images))
'-----------------------------------1. load data END-------------------------------------'

# img:  (3, x_dim, y_dim)

'-----------------------------------2. train model BEGIN-------------------------------------'
model = Default_Net_CIFAR().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train(model, optimizer, criterion, device, train_loader)

PATH = '../models/cifar_net.pth'
torch.save(model.state_dict(), PATH)
'-----------------------------------2. train model END------------------------------------'















































