import matplotlib.pyplot as plt
import numpy as np



#some parameters to be set
#epsilion
epsilons = [0, .05, .1, .15, .2, .25, .3]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# examples = np.load('../results/example_mnist_cnn_FGSM.npy', allow_pickle=True)
examples = np.load('../results/example_cifar10_d_2_FGSM.npy', allow_pickle=True)
examples = examples.tolist()
'''in the part we will demonstrate the performance of attakc models'''
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(classes[orig], classes[adv]))
        # img = ex / 2 + 0.5
        plt.imshow(np.transpose(ex, (1, 2, 0)))
plt.tight_layout()
plt.show()

