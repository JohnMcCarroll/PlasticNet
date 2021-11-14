import torch
import torch.nn as nn
import torch.nn.functional as F

### The Block class contains a stage (a series of convolutions).

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, 3, 1, 1)
        self.conv2 = nn.Conv2d(9, 9, 3, 1, 1)
        self.conv3 = nn.Conv2d(9, 3, 3, 1, 1)

    def forward(self, t):
        # expect t to be a list of incoming tensors
        # vet empty lists
        if len(t) > 0:
            shape = t[0].shape
            # vet inactive connections (1D tensors)
            if len(shape) > 1:
                out = torch.zeros(shape[0], shape[1], shape[2], shape[3])
                # residual addition of inputs
                for input in t:
                    out += input

                ## convolve
                # input layer
                # t = t

                # conv1 layer
                out = F.relu(out)
                out = self.conv1(out)
                out = F.relu(out)

                # conv2 layer
                out = self.conv2(out)
                out = F.relu(out)

                # conv3 layer
                out = self.conv3(out)
                
                return out

        # inactive state
        return 0

    # def infer(self, t):
    #     # expect t to be a list of incoming tensors
    #     if len(t) > 0:
    #         out = nn.zeros(3, 32, 32)
    #         # residual addition of inputs
    #         for input in t:
    #             out += input
    #
    #         # convolve
    #         return self.forward(out)
    #
    #     # inactive state
    #     else:
    #         return 0




# add batchnorm
# add node layer construction
# experiment with identity skips
# add ResNeXt functionality