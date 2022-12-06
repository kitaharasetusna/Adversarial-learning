import torch.nn as nn
import torch.nn.functional as F
from attackModels import *

'''we put network and traning and testing tool function in this part'''


class Default_Net(nn.Module):
    def __init__(self):
        super(Default_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def parseModel(model):
    if model=='FGSM':
        return fgsm_attack

def test(model, attack_model, device, test_loader, epsilon):
    attack = parseModel(attack_model)
    # count the accuracy
    correct = 0
    #
    adv_examples = []

    # enumerate all dataset
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # for visualization: get 5 examples of right guess after perturbation
            if (epsilon == 0) and (len(adv_examples) < 5):
                '''TODO: change device'''
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                print('adv_ex.shape ', adv_ex.shape)
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # for visualization: get 5 examples of false guess after perturbation
            if len(adv_examples) < 5:
                '''TODO: change device'''
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                print('adv_ex.shape ', adv_ex.shape)
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
