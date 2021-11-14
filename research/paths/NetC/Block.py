import torch
import torch.nn as nn
import torch.nn.functional as F

### The Block class contains a stage (a series of convolutions). These convolutions will

class Block(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck):
        super(Block, self).__init__()
        if not bottleneck:
            self.conv1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
            self.conv3 = nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        else:
            self.conv1 = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
            self.conv2 = nn.Conv2d(input_channels, output_channels, 3, 1, 2)
            self.conv3 = nn.Conv2d(output_channels, output_channels, 1, 1, 0)

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




# add batchnorm
# add node layer construction
# experiment with identity skips
# add ResNeXt functionality