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
            nn.ReLU(inplace=True)
        )

    def forward(self, t):
        # expect t to be a list of incoming tensors (curated by layer)
        # vet empty lists
        if len(t) > 0:
            out = None
            # residual addition of inputs
            for input in t:
                shape = input.shape
                # vet inactive connections (1D tensors)
                if len(shape) > 1:
                    # first time through
                    if out is None:
                        out = input
                    # every other time through
                    else:
                        out += input                                # elemwise addition?

            ## convolve
            # vet all connections inactive 
            if out is not None:
                out = self.block(out)
                return out

        # inactive state
        return 0