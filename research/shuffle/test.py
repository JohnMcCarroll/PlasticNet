import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import gc

import ResNet
import Network


### The Gymnasium script loads or initializes a PlasticNet model and trains it for a specified number of epochs on a specified dataset
## setup
# initialize hyperparameters & variables
trail_desc = '1_e1_plastic'
batchSize = 100
learningRate = 0.0001
epoch = 1
#test_set_size = 10000
dataset = list()
test_set = list()

# loading in network and data
# Creates new network
network = Network.Network(3, 10, 15).cuda()
resnet = ResNet.resnet50(3, 10).cuda()

# create mock batch
input = torch.Tensor(100, 3, 224, 224).cuda()
out = network(input)
out2 = resnet(input)






#     # init optimizer
# optimizer = optim.Adam(network.parameters(), learningRate, weight_decay=0.00001)                    # Need to find good weight decay rate [might want to only decay scalars]

# # organize data

#     # train data
# train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
# #train_losses = list()

#     # setting up test data
# test_loader = torch.utils.data.DataLoader(test_set, int(len(test_set) / 10))
# test_pics, test_results = next(iter(test_loader))
# test_results = test_results.cuda()     #switch to gpu
# test_pics = test_pics.cuda()
# test_losses = list()
# test_accuracies = list()

# torch.cuda.synchronize()

# ## training loop

# # add loops for testing hyperparams / architectures
# with torch.autograd.set_detect_anomaly(True):
#     for epoch in range(epoch):
#         print("epoch " + str(epoch))
#         count = 0
#         for batch in train_loader:

#             pics, labels = batch

#             # converting type & reshaping
#             labels = labels.cuda()       #switch to gpu
#             pics = pics.cuda()
#             torch.cuda.synchronize()

#             # calculating loss
#             preds = network(pics)
#             loss = F.cross_entropy(preds, labels)

#             #train_losses.append(loss.item())    # store train loss for batch

#             # calculating gradients
#             optimizer.zero_grad()   #clear out accumulated gradients
#             loss.backward()
#             optimizer.step()        # updating weights

#             # calculate and store test accuracy
#             test_preds = network(test_pics)
#             test_loss = F.cross_entropy(test_preds, test_results)
#             test_losses.append(test_loss.item())

#             _, predicted = torch.max(test_preds, 1) 
#             correct = 100 * ((predicted == test_results).sum().item() / len(test_results))
#             test_accuracies.append(correct)
#             print("Test Accuracy: " + str(correct))

#             # rewire network
#             count += 1
#             if count % 10 == 0:
#                 network.populate(prune_on=True)
#                 break


# ## plotting and saving training data

# plt.plot(test_losses)
# plt.ylabel('test loss')
# plt.xlabel('batch number')
# plt.savefig(r'D:\Machine Learning\PlasticNet\research\paths\outputs\testloss_' + trail_desc + '.png')
# #plt.show()

# # plt.plot(train_losses)
# # plt.ylabel('train loss')
# # plt.xlabel('prediction number')
# #plt.show()

# plt.plot(test_accuracies)
# plt.ylabel('test accuracy')
# plt.xlabel('prediction number')
# plt.savefig(r'D:\Machine Learning\PlasticNet\research\paths\outputs\testacc_' + trail_desc + '.png')


# ## save network
# torch.save(network, r'D:\Machine Learning\PlasticNet\research\paths\networks\Plastic1.0')