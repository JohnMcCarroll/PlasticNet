import torch
import torch.nn as nn
import torch.nn.functional as F
import sortedcollections
import random
import ResNet

# Lookup function for retreiving various activation functions
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

# The Fully Connected capstone to ResNet, outputing class prediction probabilities
class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

# Base class for a Residual Block (Identity placeholder in self.blocks to be overridden by child classes)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        #residual = x
        #if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        #x += residual
        x = self.activate(x)
        return x
    
    # @property
    # def should_apply_shortcut(self):
    #     return self.in_channels != self.out_channels

# ResNet Bottleneck block used to increase depth while reducing dimensionality. 1x1-3x3-1x1 with BN-relu after each.
class ResNetBottleNeckBlock(ResNet.ResNetResidualBlock):
    def __init__(self, in_channels, out_channels, inputIndex, downsampling, depth, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.inputIndex = inputIndex
        self.downsampling = downsampling

        if depth == 1:
            self.blocks = nn.Sequential(
                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
            )
        elif depth == 2:
            self.blocks = nn.Sequential(
                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
            )
        elif depth == 3:
            self.blocks = nn.Sequential(
                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
            )
        elif depth == 4:
            self.blocks = nn.Sequential(
                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
                activation_func(self.activation),

                ResNet.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
                activation_func(self.activation),
                ResNet.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
            )

        self.scalar = nn.parameter.Parameter(torch.Tensor([random.random()]))

    def forward(self, x):
        x = self.blocks(x)
        x = self.activate(x)
        x = x * self.scalar
        return x

    def get_scalar(self):
        return self.scalar.item()

    def get_inputIndex(self):
        return self.inputIndex + 1              # add one to offset for input layer


### The Network class coordinates higher level architecture search
### ResNet50 - Plastic

class Network(nn.Module):
    def __init__(self, in_channels, n_classes, maxBlocks, prune_method, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], activation='relu'):
        super(Network, self).__init__()
        # declare path
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.linear = ResnetDecoder(2048, n_classes)
        self.identity = nn.Identity()
        self.identity_bottleneck1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),                 # padding added to limit dim reduction (CIFAR)
            nn.BatchNorm2d(256)
        )
        self.identity_bottleneck2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.identity_bottleneck3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.identity_bottleneck4 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        # quick and dirty way to allow shuffling of blocks   TODO: allow for block insertion and deletion
        self.identityPath = nn.ModuleList([
            # downsampling gate
            self.gate,
            # 64 input - 256 output
            self.identity_bottleneck1, #1
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            # 256 input - 512 output
            self.identity_bottleneck2, #8
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            # 512 input - 1024 output
            self.identity_bottleneck3, #18
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            # 1024 input - 2048 output
            self.identity_bottleneck4, #34
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            self.identity,
            # FC decoder
            self.linear
        ])

        # declare plastic block infrastructure
        self.blocks = nn.ModuleDict()                   # {'output_index': ResidualBlock}       # ResBlock will have scalar & input

        self.blocks['1'] = nn.ModuleList([ResNetBottleNeckBlock(64, int(256 / 4), 0, 2, 1).cuda()])
        self.blocks['8'] = nn.ModuleList([ResNetBottleNeckBlock(256, int(512 / 4), 7, 2, 1).cuda()])
        self.blocks['18'] = nn.ModuleList([ResNetBottleNeckBlock(512, int(1024 / 4), 17, 2, 1).cuda()])
        self.blocks['34'] = nn.ModuleList([ResNetBottleNeckBlock(1024, int(2048 / 4), 33, 2, 1).cuda()])

        # declare variables
        self.prune_method = prune_method
        self.numBlocks = 0
        self.maxBlocks = maxBlocks
        self.outputs = list()
        self.threshold = 0.5
        self.blockID = 0
        self.populate(False)

    
    def forward(self, t):
        # add raw input to outputs list
        self.outputs.clear()
        self.outputs.append(t)

        for i in range(len(self.identityPath)):
            # identity filter
            t = self.identityPath[i](t)
            # apply plasitc blocks filters
            residual = None
            if str(i) in self.blocks:
                for j, block in enumerate(self.blocks[str(i)]):
                    inputIndex = block.get_inputIndex()
                    if residual is None:
                        residual = block(self.outputs[inputIndex])
                    else:
                        residual += block(self.outputs[inputIndex])
            if residual is not None:
                t += residual
            # log output
            self.outputs.append(t.clone())          # TODO: want better performing solution***

        # return predictions
        return t

    def populate(self, prune_on):
        ## prune
        if prune_on or not prune_on:
            if self.prune_method == 'threshold':
                self.prune()
            elif self.prune_method == 'gradient':
                self.prune_gradient(prune_on)
            else:
                self.prune_lowest(prune_on)

        ## create blocks
        while self.numBlocks < self.maxBlocks:

            inputIndex = None
            outputIndex = None

            # selecting indecies
            indexA = random.randrange(0, 38)
            indexB = indexA + 1

            if indexA > indexB:
                inputIndex = indexB
                outputIndex = indexA
            elif indexA < indexB:
                inputIndex = indexA
                outputIndex = indexB
            else:
                inputIndex = indexA
                outputIndex = indexB + 1
                # if inputIndex == 0:
                #     inputIndex += 1
                #     outputIndex += 1
                # if outputIndex == 17:
                #     inputIndex -= 1
                #     outputIndex -= 1
            
            # ensure no duplicates
            if str(outputIndex) in self.blocks:
                continue

            # calculate channels 
            in_channels = None
            out_channels = None

            if inputIndex == 0:
                in_channels = 64
            elif inputIndex <= 7:
                in_channels = 256
            elif inputIndex <= 17:
                in_channels = 512
            elif inputIndex <= 33:
                in_channels = 1024
            elif inputIndex > 33:
                in_channels = 2048

            if outputIndex <= 7:
                out_channels = 256
            elif outputIndex <= 17:
                out_channels = 512
            elif outputIndex <= 33:
                out_channels = 1024
            elif outputIndex > 33:
                out_channels = 2048
            
            # calculate downsampling & depth
            depth = 1
            downsampling = 1

            span = range(inputIndex + 1, outputIndex + 1)
            bottlenecks_crossed = 0
            if 1 in span:
                bottlenecks_crossed += 1
            if 8 in span:
                bottlenecks_crossed += 1
            if 18 in span:
                bottlenecks_crossed += 1
            if 34 in span:
                bottlenecks_crossed += 1

            if bottlenecks_crossed == 0:
                pass
            elif bottlenecks_crossed == 1:
                downsampling = 2
            else:
                #depth = bottlenecks_crossed
                continue

            # get blockID
            self.blockID += 1
            ID = str(self.blockID)

            # create block
            if str(outputIndex) in self.blocks:
                self.blocks[str(outputIndex)]._modules[ID] = ResNetBottleNeckBlock(in_channels, int(out_channels / 4), inputIndex, downsampling, depth).cuda()
            else:
                self.blocks[str(outputIndex)] = nn.ModuleList([ResNetBottleNeckBlock(in_channels, int(out_channels / 4), inputIndex, downsampling, depth).cuda()])

            self.numBlocks += depth

    def prune(self):
        # prune blocks with scalar values below threshold
        numPruned = 0
        for key, block_list in self.blocks.items():
            index = -1
            for block in block_list:
                index += 1
                if block.get_scalar() < self.threshold:
                    del self.blocks[key]._modules[str(index)]
                    numPruned += 1

        with open(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\log.txt', 'a') as f:
            f.write("Pruned " + str(numPruned) + " blocks" + '\n')

    def prune_lowest(self, prune_on):
        if prune_on:
            # prune blocks with scalar values below threshold
            lowest_scalar = (None, None, 10)
            for key in self.blocks.keys():
                for index, block in self.blocks[key]._modules.items():
                    if block.get_scalar() < lowest_scalar[2]:
                        lowest_scalar = (key, index, block.get_scalar())
            #remove lowest
            if lowest_scalar[0] != None:
                modules = self.blocks[lowest_scalar[0]]._modules
                del modules[str(lowest_scalar[1])]
                self.numBlocks -= 1

                with open(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\plastic_gradients.txt', 'a') as f:
                    f.write("Pruned: " + str(lowest_scalar) + '\n')

    def prune_gradient(self, isPrune):
        # calc block gradient
        gradients = sortedcollections.SortedDict()
        for key in self.blocks.keys():
            for index, block in self.blocks[key]._modules.items():

                gradient = 0
                for Filter in block.blocks:
                    for stage in Filter._modules:
                        if Filter._modules[stage].weight.grad is not None:
                            grad = torch.abs(Filter._modules[stage].weight.grad)
                        else:
                            grad = torch.zeros(1)
                        gradient += torch.sum(grad)

                if block.shortcut:
                    for Filter in block.shortcut:
                        if Filter.weight.grad:
                            grad = torch.abs(Filter.weight.grad)
                            gradient += torch.sum(grad)
                
                gradients[gradient.item()] = (key, index)

        # print block gradient
        if not isPrune:
            with open(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\plastic_gradients.txt', 'a') as file:
                for key in gradients:
                    file.write(str(gradients[key]) + ":" + str(key))
                    file.write('\n')
                file.write('\n')

        # prune lowest
        if isPrune:
            grads = gradients.keys()
            key, index = gradients[grads[0]]

            # del self.blocks
            modules = self.blocks[key]._modules
            del modules[index]
            self.numBlocks -= 1

            with open(r'D:\Machine Learning\PlasticNet\research\shuffle\outputs\plastic_gradients.txt', 'a') as file:
                file.write("block pruned: " + key + ", " + index + '\n')

# import torch
# net = CNN()
# output = net(torch.rand([1,14,8,8]))
# print(output.size())
