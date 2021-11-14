import torch
import torch.nn as nn
import torch.nn.functional as F

### The Layer class coordinates connections between layers and maintains a layer's stages (Blocks)

class FinalLayer(nn.Module):
    def __init__(self):
        super(FinalLayer, self).__init__()

        # declare building blocks
        self.conv1 = nn.Conv2d(9, 3, 3, 2)
        self.conv2 = nn.Conv2d(3, 1, 3, 1)
        self.linear = nn.Linear(13*13, 10, True)

    def forward(self, t):
        # expect t to be a list of each previous layer's block's outputs
        # take last 3 outputs
        inputA = None
        inputB = None
        inputC = None
        shape = t[0].shape
        if type(t[-3]) is not int:
            inputA = t[-3]
        else:
            inputA = torch.zeros(shape[0], shape[1], shape[2], shape[3])

        if type(t[-2]) is not int:
            inputB = t[-2]
        else:
            inputB = torch.zeros(shape[0], shape[1], shape[2], shape[3])

        if type(t[-1]) is not int:
            inputC = t[-1]
        else:
            inputC = torch.zeros(shape[0], shape[1], shape[2], shape[3])

        t = torch.cat((inputA, inputB, inputC), 1)

        # conv1 layer
        t = F.relu(t)
        t = self.conv1(t)
        t = F.relu(t)

        # conv2 layer
        t = self.conv2(t)
        t = F.relu(t)

        # FC layer
        t = t.reshape(-1, 13 * 13)
        t = self.linear(t)
        t = F.softmax(t, 1)

        # return predictions
        return t



# Not pumped about final module. Seems rushed and idk how well info will compress

## Debug
# import torch
# input = [torch.Tensor(1, 3, 32, 32), torch.Tensor(1, 3, 32, 32), torch.Tensor(1, 3, 32, 32)]
# net = FinalLayer()
#
# out = net(input)
#
# print(out)