import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from BlockA import BlockA
from BlockAA import BlockAA
from BlockB import BlockB
from BlockBB import BlockBB
from BlockL import BlockL
from BlockC import BlockC
from BlockCC import BlockCC
from LayerA import LayerA

### The Network class coordinates higher level architecture search

class NetworkA(nn.Module):
    def __init__(self):
        super(NetworkA, self).__init__()
        # Declare Blocks
        self.blockA1 = BlockA()
        self.blockA2 = BlockA()
        self.blockA3 = BlockA()

        self.blockAA1 = BlockAA()
        self.blockAA2 = BlockAA()
        self.blockAA3 = BlockAA()

        self.blockB1 = BlockB()
        self.blockB2 = BlockB()
        self.blockB3 = BlockB()

        self.blockBB1 = BlockBB()
        self.blockBB2 = BlockBB()
        self.blockBB3 = BlockBB()

        self.blockC1 = BlockC()
        self.blockC2 = BlockC()
        self.blockC3 = BlockC()

        self.blockCC1 = BlockCC()
        self.blockCC2 = BlockCC()
        self.blockCC3 = BlockCC()

        self.linear = BlockL()

        # Layer A
        self.layerA = LayerA([self.blockA1, self.blockA2, self.blockA3])

        # Layer AA
        self.layerAA = LayerA([self.blockAA1, self.blockAA2, self.blockAA3])

        # Layer B
        self.layerB = LayerA([self.blockB1, self.blockB2, self.blockB3])

        # Layer BB
        self.layerBB = LayerA([self.blockBB1, self.blockBB2, self.blockBB3])

        # Layer C
        self.layerC = LayerA([self.blockC1, self.blockC2, self.blockC3])

        # Layer CC
        self.layerCC = LayerA([self.blockCC1, self.blockCC2, self.blockCC3])

        # Final
        self.final = LayerA([self.linear])

        # variables
        self.layers = [self.layerA, self.layerAA, self.layerB, self.layerBB, self.layerC, self.layerCC, self.final]
        self.outputs = list()

        self.connectivity = 40
        self.threshold = 0.05

        self.numConnections = 0
        # self.rewire()
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
        return self.outputs[19]                                         #hardcorded... not the best

    def rewire(self):
        # prune layers
        self.layerA.prune(self.threshold)                  #3
        self.layerAA.prune(self.threshold)
        self.layerB.prune(self.threshold)                  #6
        self.layerBB.prune(self.threshold)
        self.layerC.prune(self.threshold)                  #9
        self.layerCC.prune(self.threshold)
        self.final.prune(self.threshold)                  #10

        # redistribute connections randomly
        self.numConnections = self.countConnections()
        amount = self.connectivity - self.numConnections
        if amount > 0:
            print("Connections Rewired: " + str(amount))

        while amount > 0:
            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.final.wire(numNew, [18, 17, 16])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerCC.wire(numNew, [15, 14, 13]) #, 12, 11, 10])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerC.wire(numNew, [12, 11, 10])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerBB.wire(numNew, [9, 8, 7]) #, 6, 5, 4])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerB.wire(numNew, [6, 5, 4])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerAA.wire(numNew, [3, 2, 1]) #, 0])

            numNew = random.randrange(0, amount + 1)
            amount -= numNew
            self.layerA.wire(numNew, [0])

            self.numConnections = self.countConnections()
            amount = self.connectivity - self.numConnections

    def countConnections(self):
        amount = 0
        amount += self.layerA.countConnections()
        amount += self.layerB.countConnections()
        amount += self.layerC.countConnections()
        amount += self.layerAA.countConnections()
        amount += self.layerBB.countConnections()
        amount += self.layerCC.countConnections()
        amount += self.final.countConnections()
        return amount

    def baselineWire(self):
        self.layerA.baselineWire(1)
        self.layerB.baselineWire(2)
        self.layerC.baselineWire(3)
        self.layerAA.baselineWire(5)
        self.layerBB.baselineWire(6)
        self.layerCC.baselineWire(7)
        self.final.baselineWire(4)


# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())