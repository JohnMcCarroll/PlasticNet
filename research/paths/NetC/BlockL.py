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

                ## flatten
                out = out.view(out.size(0), -1)

                ## convolve
                out = self.block(out)
                
                return out

        # inactive state
        return 0