import torch
import torch.nn as nn
import torch.nn.functional as F

### The Block class contains a stage (a series of convolutions). These convolutions will

class BlockL(nn.Module):
    def __init__(self):
        super(BlockL, self).__init__()
        self.block = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
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
                # flatten
                out = out.view(out.size(0), -1)
                out = self.block(out)
                return out

        # inactive state
        return 0