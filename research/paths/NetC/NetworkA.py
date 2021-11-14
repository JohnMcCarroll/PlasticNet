import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import Layer
from FinalLayer import FinalLayer
import random
from BlockA import BlockA
from BlockB import BlockB
from BlockC import BlockC
from BlockL import BlockL
from LayerA import LayerA

### The Network class coordinates higher level architecture search

class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        # Declare Blocks
        self.blockA1 = BlockA()
        self.blockA2 = BlockA()
        self.blockA3 = BlockA()
        self.blockB1 = BlockB()
        self.blockB2 = BlockB()
        self.blockB3 = BlockB()
        self.blockC1 = BlockC()
        self.blockC2 = BlockC()
        self.blockC3 = BlockC()
        self.linear = BlockL()

        # Layer A
        self.layerA = LayerA([self.blockA1, self.blockA2, self.blockA3])

        # Layer B
        self.layerB = LayerA([self.blockB1, self.blockB2, self.blockB3])

        # Layer C
        self.layerC = LayerA([self.blockC1, self.blockC2, self.blockC3])

        # Final
        self.final = LayerA([self.linear])

        # variables
        self.layers = [self.layerA, self.layerB, self.layerC, self.final]
        self.outputs = list()

        self.connectivity = 60
        self.threshold = 0.05

        self.numConnections = 0
        #self.rewire()
        self.baselineWire()
    
    def forward(self, t):
        # add raw input to outputs list
        self.outputs.clear()

        # initial processing
        self.outputs.append(t)

        # pass through layers
        for layer in self.layers:
            self.outputs.extend(layer(self.outputs))           # add outputs of current layer to list of outputs

        # return predictions (should be last entry to outputs...)
        return self.outputs[-1]

    def rewire(self):
        # prune layers
        self.layer1a.prune(self.threshold)                  #3
        self.layer1b.prune(self.threshold)                  #6
        self.layerB1.prune(self.threshold)                  #9
        self.layer2a.prune(self.threshold)                  #12
        self.layer2b.prune(self.threshold)                  #15
        self.layerB2.prune(self.threshold)                  #18
        self.layer3a.prune(self.threshold)                  #21
        self.layer3b.prune(self.threshold)                  #24
        self.layerB3.prune(self.threshold)                  #27
        self.layer4.prune(self.threshold)                   #30

        # redistribute connections randomly
        self.numConnections = self.countConnections()
        amount = self.connectivity - self.numConnections

        while amount > 0:
            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer4.wire(numNew, [27, 26, 25])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerB3.wire(numNew, [24, 23, 22, 21, 20, 19, 18, 17, 16])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer3b.wire(numNew, [21, 20, 19, 18, 17, 16])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer3a.wire(numNew, [18, 17, 16])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerB2.wire(numNew, [15, 14, 13, 12, 11, 10, 9, 8, 7])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer2b.wire(numNew, [12, 11, 10, 9, 8, 7])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer2a.wire(numNew, [9, 8, 7])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerB1.wire(numNew, [6, 5, 4, 3, 2, 1])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer1b.wire(numNew, [3, 2, 1])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layer1a.wire(numNew, [0])

            self.numConnections = self.countConnections()
            amount = self.connectivity - self.numConnections

    def countConnections(self):
        amount = 0
        amount += self.layer1.countConnections()
        amount += self.layer2.countConnections()
        amount += self.layer3.countConnections()
        return amount

    def baselineWire(self):
        self.layerA.baselineWire(1)
        self.layerB.baselineWire(2)
        self.layerC.baselineWire(3)
        self.final.baselineWire(4)


# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())