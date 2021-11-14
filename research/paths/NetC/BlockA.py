import torch
import torch.nn as nn
import torch.nn.functional as F

### The Block class contains a stage (a series of convolutions). These convolutions will

class BlockA(nn.Module):
    def __init__(self):
        super(BlockA, self).__init__()
        self.block = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

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
                out = self.block(out)
                
                return out

        # inactive state
        return 0