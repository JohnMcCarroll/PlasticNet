import torch
import torch.nn as nn
import torch.nn.functional as F
from Block import Block
import random

### The Layer class coordinates connections between layers and maintains a layer's stages (Blocks)

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

        # declare building blocks
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()

        # declare connections dict
        self.connections = dict()               # {'block index':[upstream indices]}
        self.connections['1'] = []
        self.connections['2'] = []
        self.connections['3'] = []

        # declare connection scalars
        self.scalars = nn.ParameterDict()       # {'block+input':param}

    def forward(self, t):
        # expect t to be a list of each previous layer's block's outputs
        out = list()

        # gather outputs                                                            # change to key loop
        input = list()
        for inputIndex in self.connections['1']:
            input.append(t[inputIndex] * self.scalars['1' + str(inputIndex)])
        out.append(self.block1(input))

        input.clear()
        for inputIndex in self.connections['2']:
            input.append(t[inputIndex] * self.scalars['2' + str(inputIndex)])
        out.append(self.block2(input))

        input.clear()
        for inputIndex in self.connections['3']:
            input.append(t[inputIndex] * self.scalars['3' + str(inputIndex)])
        out.append(self.block3(input))

        return out

    def prune(self, threshold):
        # iterate through scalars and remove connections less than threshold
        for key in list(self.scalars.keys()):
            if abs(self.scalars[key]) < threshold:
                self.scalars.pop(key)
                blocksConnections = self.connections[key[0]]
                blocksConnections.remove(int(key[1:]))                # remove upstream index from connection list

    def wire(self, numNew, numUpstream):
        # attempts to make designated amount of connections - no guarentee
        while numNew > 0:
            upstream = random.randint(0, numUpstream - 1)
            downstream = random.randint(1, 3)                                           # TODO: remove hardcode

            # ensure connection not redundant
            blocksConnections = self.connections[str(downstream)]
            if upstream in blocksConnections:
                numNew -= 1
                continue
            else:
                # create new connection
                blocksConnections.append(upstream)
                key = str(downstream) + str(upstream)
                self.scalars[key] = nn.parameter.Parameter(torch.Tensor([random.random()]))        # might want wider range

            numNew -= 1

    def countConnections(self):
        return len(self.scalars)

    # def getActivities(self):
    #     activity = [False, False, False]
    #     if '1' in self.connections:
    #         activity[0] = True
    #     if '2' in self.connections:
    #         activity[1] = True
    #     if '3' in self.connections:
    #         activity[2] = True


# give connections initial state
# change to orderedDict for convenient key loop
# implement rewire (create new block if first connection)