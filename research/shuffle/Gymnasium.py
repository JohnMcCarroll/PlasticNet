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
import NetworkB


### The Gymnasium script loads or initializes a PlasticNet model and trains it for a specified number of epochs on a specified dataset
# assay of tests
learning_rate = [0.001, 0.0003, 0.0001]
networks = ['resnet', 'a', 'b']
weight_decay = [0.0]
prune_method = ['lowest', 'gradient']   #lowest_gradient?\
prune_freq = [500, 1000]

logfile_path = r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\logPlasticGRAD.txt'


for lr in learning_rate:
    for net in networks:
        # toggle network types
        if net == 'resnet' or net == 'a':
            prune_method = ['none']
            prune_freq = [1]
        if net == 'b':
            prune_method = ['lowest', 'gradient']
            prune_freq = [500, 1000]

        for pm in prune_method:
            for pf in prune_freq:
                ## setup
                # initialize hyperparameters & variables
                trial_desc = net + '50_lr' + str(lr) + '_pm' + pm
                batchSize = 50
                learningRate = lr
                epoch = 10
                #test_set_size = 10000
                dataset = list()
                test_set = list()

                isPrune = True
                if pm == 'none':
                    isPrune = False

                # loading in network and data
                # Creates new network
                network = None

                if net == 'resnet':
                    continue
                    network = ResNet.resnet50(3, 10).cuda()
                if net == 'a':
                    continue
                    network = Network.Network(3, 10, 15, pm).cuda()
                if net == 'b':
                    network = NetworkB.Network(3, 10, 15, pm).cuda()
                
                ### continue where left off ###
                if net == 'resnet' and lr == 0.001:
                    continue
                if net == 'a' and lr == 0.001:
                    continue

                    # retrieve dataset
                # CIFAR
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

                # IMAGENET
                # train_set = torchvision.datasets.ImageNet(
                #     root=r'D:\Machine Learning\PlasticNet\research\paths\data',
                #     split='train',
                #     download=True, 
                #     transform=transforms.Compose([ 
                #         transforms.ToTensor() 
                #         ])
                #     )

                # test_set = torchvision.datasets.ImageNet(
                #     root=r'D:\Machine Learning\PlasticNet\research\paths\data',
                #     split='val',
                #     download=True,
                #     transform=transforms.Compose([
                #         transforms.Resize(224,224),
                #         transforms.ToTensor()
                #     ]))

                    # init optimizer
                optimizer = optim.Adam(network.parameters(), learningRate) #, weight_decay=wd)                    # Need to find good weight decay rate [might want to only decay scalars]

                # organize data

                    # train data
                train_loader = torch.utils.data.DataLoader(train_set, batchSize, shuffle=True)
                #train_losses = list()

                    # setting up test data
                test_loader = torch.utils.data.DataLoader(test_set, int(len(test_set) / 200))
                # test_pics, test_results = next(iter(test_loader))
                # test_results = test_results.cuda()     #switch to gpu
                # test_pics = test_pics.cuda()
                test_losses = list()
                test_accuracies = list()

                #torch.cuda.synchronize()

                ## training loop

                with open(logfile_path, 'a') as f:
                    f.write(trial_desc + '\n')

                # add loops for testing hyperparams / architectures
                for epoch in range(epoch):
                    print("epoch " + str(epoch))
                    with open(logfile_path, 'a') as f:
                        f.write('epoch' + str(epoch) + '\n')
                    count = 0
                    for batch in train_loader:

                        pics, labels = batch

                        # converting type & reshaping
                        labels = labels.cuda()       #switch to gpu
                        pics = pics.cuda()
                        torch.cuda.synchronize()

                        # calculating loss
                        preds = network(pics)
                        loss = F.cross_entropy(preds, labels)

                        #train_losses.append(loss.item())    # store train loss for batch

                        # calculating gradients
                        optimizer.zero_grad()   #clear out accumulated gradients
                        loss.backward()
                        optimizer.step()        # updating weights


                        # calculate and store test accuracy & gradients
                        if count % 30 == 0:
                            # calc gradients
                            if isPrune:
                                network.populate(prune_on=False)

                            # accuracy test
                            test_loss = 0
                            num_correct = 0
                            test_predictions = []
                            with torch.no_grad():
                                for test_batch in test_loader:
                                    test_pics, test_results = test_batch
                                    test_pics = test_pics.cuda()
                                    test_results = test_results.cuda()
                                    torch.cuda.synchronize()

                                    test_preds = network(test_pics)
                                    test_loss += F.cross_entropy(test_preds, test_results)

                                    _, predicted = torch.max(test_preds, 1) 
                                    num_correct += (predicted == test_results).sum().item()
                                
                                test_losses.append(test_loss.item())
                                correct = 100 * (num_correct / len(test_set))
                                test_accuracies.append(correct)
                                print("Test Accuracy: " + str(correct))

                                # write log
                                with open(logfile_path, 'a') as f:
                                    f.write('accuracy: ' + str(correct))
                                    f.write('\n')

                        # rewire network
                        if isPrune and count % pf == 0 and epoch < 9:
                            network.populate(prune_on=True)

                        count += 1


                ## plotting and saving training data
                plt.clf()
                plt.plot(test_losses)
                plt.ylabel('test loss')
                plt.xlabel('batch number')
                plt.savefig(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\testloss_' + trial_desc + '.png')
                #plt.show()

                # plt.plot(train_losses)
                # plt.ylabel('train loss')
                # plt.xlabel('prediction number')
                #plt.show()

                plt.clf()
                plt.plot(test_accuracies)
                plt.ylabel('test accuracy')
                plt.xlabel('prediction number')
                plt.savefig(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\testacc_' + trial_desc + '.png')

                # log run info
                with open(logfile_path, 'a') as f:
                    f.write('accuracy: ' + str(test_accuracies) + '\n')
                    f.write('losses: ' + str(test_accuracies) + '\n')
                    f.write('\n')
                

                ## save network
                torch.save(network, r"D:\\Machine Learning\\PlasticNet\\research\\shuffle\\networks\\" + trial_desc)

# write out run data to a log file