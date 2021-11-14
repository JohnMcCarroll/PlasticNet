import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from Network import Network
import matplotlib.pyplot as plt
import gc

from CNN import CNN
from NetworkA import NetworkA


### The Gymnasium script loads or initializes a PlasticNet model and trains it for a specified number of epochs on a specified dataset
## setup
# initialize hyperparameters & variables
batchSize = 50
learningRate = 0.0001
epoch = 1
#test_set_size = 10000
dataset = list()
test_set = list()

# loading in network and data
# Creates new network
###network = Network()#.cuda()
network = NetworkA()

    # retrieve dataset
train_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]))

test_set = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ]))

# sanity checks & intel
# for param in network.named_parameters():
#     print(param)
# print(len(train_set))
# print(len(test_set))
# print(train_set[0])
# print((test_set[0][0].shape))
# print(train_set.classes)
# print(len(train_set.classes))

    # init optimizer
optimizer = optim.Adam(network.parameters(), learningRate, weight_decay=0.00001)                    # Need to find good weight decay rate [might want to only decay scalars]

# organize data

    # train data
train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
train_losses = list()

    # setting up test data
test_loader = torch.utils.data.DataLoader(test_set, int(len(test_set) / 250))
test_pics, test_results = next(iter(test_loader))
#test_results = test_results.float().reshape([-1, 1]).cuda()     #switch to gpu
#test_pics = test_pics.cuda()
test_losses = list()



## training loop

#add loops for testing hyperparams / architectures

for epoch in range(epoch):
    count = 0
    for batch in train_loader:

        pics, labels = batch

        # converting type & reshaping
        #labels = labels.reshape([-1, 1])#.cuda()       #switch to gpu
        #pics = pics#.cuda()

        # calculating loss
        preds = network(pics)
        loss = F.cross_entropy(preds, labels)

        train_losses.append(loss.item())    # store train loss for batch

        # calculating gradients
        optimizer.zero_grad()   #clear out accumulated gradients
        loss.backward()
        optimizer.step()        # updating weights

        # benchmark if learning
        test_preds = network(test_pics)
        test_loss = F.cross_entropy(test_preds, test_results)
        test_losses.append(test_loss.item())

        # rewire network
        # count += 1
        # if count > 100:
        #     break
        # if count % 5 == 0:
        #network.rewire()


plt.plot(test_losses)
plt.ylabel('test loss')
plt.xlabel('batch number')
plt.show()

plt.plot(train_losses)
plt.ylabel('train loss')
plt.xlabel('prediction number')
plt.show()

# save network
torch.save(network, r'D:\Machine Learning\PlasticNet\research\sequential\networks\baselineNet0.1')