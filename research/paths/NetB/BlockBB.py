import torch
import torch.nn as nn
import torch.nn.functional as F

### The Block class contains a stage (a series of convolutions). These convolutions will

class BlockBB(nn.Module):
    def __init__(self):
        super(BlockBB, self).__init__()
        self.block = nn.Sequential(
            # conv block 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
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