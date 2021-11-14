import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import Layer
from FinalLayer import FinalLayer
import random

### The Network class coordinates higher level architecture search

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layer1 = Layer()
        self.layer2 = Layer()
        self.layer3 = Layer()
        self.final = FinalLayer()

        self.outputs = list()

        self.connectivity = 20
        self.threshold = 0.05

        self.numConnections = 0
        self.rewire()
    
    def forward(self, t):
        # add raw input to outputs list
        self.outputs.clear()
        self.outputs.append(t)

        # pass to layer 1
        layer1outputs = self.layer1(self.outputs)
        for output in layer1outputs:
            self.outputs.append(output)

        # pass to layer 2
        layer2outputs = self.layer2(self.outputs)
        for output in layer2outputs:
            self.outputs.append(output)

        # pass to layer 3
        layer3outputs = self.layer3(self.outputs)
        for output in layer3outputs:
            self.outputs.append(output)

        # pass to final conv & linear layers
        t = self.final(self.outputs)

        # return predictions
        return t

    def rewire(self):
        # prune layers
        self.layer1.prune(self.threshold)
        self.layer2.prune(self.threshold)
        self.layer3.prune(self.threshold)

        # redistribute connections randomly
        self.numConnections = self.countConnections()
        amount = self.connectivity - self.numConnections

        while amount > 0:
            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer3.wire(numNew, 7)

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer2.wire(numNew, 4)

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer1.wire(numNew, 1)

            self.numConnections = self.countConnections()
            amount = self.connectivity - self.numConnections

    def countConnections(self):
        amount = 0
        amount += self.layer1.countConnections()
        amount += self.layer2.countConnections()
        amount += self.layer3.countConnections()
        return amount



# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())


