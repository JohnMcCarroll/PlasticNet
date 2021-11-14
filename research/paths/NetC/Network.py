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

        # Initial Processing
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.pool = nn.MaxPool2d(3, 2)

        # Stage 1
        self.layer1a = Layer(64, 64)
        self.layer1b = Layer(64, 64)

        # Bottleneck 1
        self.layerB1 = Layer(64, 128, True)

        # Stage 2
        self.layer2a = Layer(128, 128)
        self.layer2b = Layer(128, 128)

        # Bottleneck 2
        self.layerB2 = Layer(128, 256, True)

        # Stage 3
        self.layer3a = Layer(256, 256)
        self.layer3b = Layer(256, 256)

        # Bottleneck 3
        self.layerB3 = Layer(256, 512, True)

        # Stage 4
        self.layer4 = Layer(512, 512)

        # Final
        self.final = FinalLayer()

        # variables
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
        t = self.conv1(t)
        t = self.pool(t)
        self.outputs.append(t)

        # pass to layer 1a
        layer1aoutputs = self.layer1a(self.outputs)
        for output in layer1aoutputs:
            self.outputs.append(output)

        # pass to layer 1b
        layer1boutputs = self.layer1b(self.outputs)
        for output in layer1boutputs:
            self.outputs.append(output)

        # pass to bottleneck 1
        layerB1outputs = self.layerB1(self.outputs)
        for output in layerB1outputs:
            self.outputs.append(output)

        # pass to layer 2a
        layer2aoutputs = self.layer2a(self.outputs)
        for output in layer2aoutputs:
            self.outputs.append(output)

        # pass to layer 2b
        layer2boutputs = self.layer2b(self.outputs)
        for output in layer2boutputs:
            self.outputs.append(output)

        # pass to bottleneck 2
        layerB2outputs = self.layerB2(self.outputs)
        for output in layerB2outputs:
            self.outputs.append(output)

        # pass to layer 3a
        layer3aoutputs = self.layer3a(self.outputs)
        for output in layer3aoutputs:
            self.outputs.append(output)

        # pass to layer 3b
        layer3boutputs = self.layer3b(self.outputs)
        for output in layer3boutputs:
            self.outputs.append(output)

        # pass to bottleneck 3
        layerB3outputs = self.layerB3(self.outputs)
        for output in layerB3outputs:
            self.outputs.append(output)

        # pass to layer 4
        layer4outputs = self.layer4(self.outputs)
        for output in layer4outputs:
            self.outputs.append(output)

        # pass to final conv & linear layers
        t = self.final(self.outputs)

        # return predictions
        return t

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
        self.layer1a.baselineWire(1)
        self.layer1b.baselineWire(2)
        self.layerB1.baselineWire(3)
        self.layer2a.baselineWire(4)
        self.layer2b.baselineWire(5)
        self.layerB2.baselineWire(6)
        self.layer3a.baselineWire(7)
        self.layer3b.baselineWire(8)
        self.layerB3.baselineWire(9)
        self.layer4.baselineWire(10)


# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())


