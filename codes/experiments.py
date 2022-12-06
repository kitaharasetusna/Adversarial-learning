#regular import for pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from models import *

import  matplotlib.pyplot as plt

#some parameters to be set
'''TODO: add argparse'''
#default model
pretrained_model = "../models/lenet_mnist_model.pth"
#whether to use cuda
use_cuda=True
#epsilion
epsilons = [0, .05, .1, .15, .2, .25, .3]
#name of attack model
ATTACK_MODEL = 'FGSM'
'''TODO: make it work on cifar'''
DATASET = 'mnist'
TARGET_MODEL = "cnn"

'''----------------------------0.initialize pytorch BEGIN----------------------------------'''
#initialization device to use
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
'''----------------------------0.initialize pytorch  END----------------------------------'''

'''----------------------------1.load the trained model BEGIN----------------------------------'''
model = Default_Net().to(device)
#Pytorch doc:
# torch.load() uses Python’s unpickling facilities
# but treats storages, which underlie tensors, specially.
# They are first deserialized on the CPU and are then moved
# to the device they were saved from.
# However, storages can be dynamically remapped to
#an alternative set of devices using the map_location argument.
'''TODO1: change this into cuda'''
model.load_state_dict(torch.load(pretrained_model, map_location='cuda'))


# ---------------------about how to save and load model
# torch.save(model.state_dict(), PATH)
# device = torch.device('cpu')
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH, map_location=device))

#model.eval() is a kind of switch for some specific layers/parts
# of the model that behave differently during training and inference
# (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.
# You need to turn off them during model evaluation, and .eval() will do it for you.
model.eval()
'''----------------------------1.load the trained model END---------------------------------'''
# evaluate model:
# model.eval()
#
# with torch.no_grad():
#     ...
#     out_data = model(data)

# training step
# ...
# model.train()
# ...
'''----------------------------2.load the dataset BEGIN----------------------------------'''
accuracies = []
examples = []
'''
TODO: doc mention that if it's first time, you have to set download option True
'''
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
# Run test for each epsilon

'''----------------------------2.load the dataset END ---------------------------------------'''

'''-------------------------- -3.testing the attack BEGIN---------------------------------'''
epoch_cnt = 0
for eps in epsilons:
    acc, ex = test(model, ATTACK_MODEL, device, test_loader, eps)
    if(epoch_cnt==0):
        print('len(ex)', len(ex))
    epoch_cnt=epoch_cnt+1
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

'''-------------------------- -3.testing the attack END---------------------------------'''


'''-------------------------- -4.save exmaples BEGIN---------------------------------'''
examples = np.array(examples)
np.save('../results/example_{0}_{2}_{1}'.format(DATASET, ATTACK_MODEL, TARGET_MODEL), examples)
