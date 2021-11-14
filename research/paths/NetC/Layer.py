import torch
import torch.nn as nn
import torch.nn.functional as F
from Block import Block
import random

### The Layer class coordinates connections between layers and maintains a layer's stages (Blocks)

class Layer(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck=False):
        super(Layer, self).__init__()
        # declare building blocks
        self.block1 = Block(input_channels, output_channels, bottleneck)
        self.block2 = Block(input_channels, output_channels, bottleneck)
        self.block3 = Block(input_channels, output_channels, bottleneck)

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

    def wire(self, numNew, upstream):
        # attempts to make designated amount of connections - no guarentee
        while numNew > 0:
            numNew -= 1

            upstream = random.randint(0, upstream - 1)
            downstream = random.randint(1, 3)                                           # might want to not hardcode?

            # ensure connection not redundant
            blocksConnections = self.connections[str(downstream)]
            if upstream in blocksConnections:
                continue
            else:
                # create new connection
                blocksConnections.append(upstream)
                key = str(downstream) + str(upstream)
                self.scalars[key] = nn.parameter.Parameter(torch.Tensor([random.random()]))        # might want wider range

    def countConnections(self):
        return len(self.scalars)

    def baselineWire(self, position):
        if position == 1:
            self.connections['1'].append(0)
            self.connections['2'].append(0)
            self.connections['3'].append(0)
            self.scalars['10'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['20'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['30'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 2:
            self.connections['1'].append(1)
            self.connections['2'].append(2)
            self.connections['3'].append(3)
            self.scalars['11'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['22'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['33'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 3:
            self.connections['1'].append(4)
            self.connections['2'].append(5)
            self.connections['3'].append(6)
            self.scalars['14'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['25'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['36'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 4:
            self.connections['1'].append(7)
            self.connections['2'].append(8)
            self.connections['3'].append(9)
            self.scalars['17'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['28'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['39'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 5:
            self.connections['1'].append(10)
            self.connections['2'].append(11)
            self.connections['3'].append(12)
            self.scalars['110'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['211'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['312'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 6:
            self.connections['1'].append(13)
            self.connections['2'].append(14)
            self.connections['3'].append(15)
            self.scalars['113'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['214'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['315'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 7:
            self.connections['1'].append(16)
            self.connections['2'].append(17)
            self.connections['3'].append(18)
            self.scalars['116'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['217'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['318'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 8:
            self.connections['1'].append(19)
            self.connections['2'].append(20)
            self.connections['3'].append(21)
            self.scalars['119'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['220'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['321'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 9:
            self.connections['1'].append(22)
            self.connections['2'].append(23)
            self.connections['3'].append(24)
            self.scalars['122'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['223'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['324'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
        if position == 10:
            self.connections['1'].append(25)
            self.connections['2'].append(26)
            self.connections['3'].append(27)
            self.scalars['125'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['226'] = nn.parameter.Parameter(torch.Tensor([random.random()]))
            self.scalars['327'] = nn.parameter.Parameter(torch.Tensor([random.random()]))



# give connections initial state
# change to orderedDict for convenient key loop
# implement rewire (create new block if first connection)