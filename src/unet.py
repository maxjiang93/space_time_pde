import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils import *



class UNet(nn.Module):

    """
    UNet model implementation:
        Architecture follows the one in paper: https://arxiv.org/abs/1505.04597.
        U-Net is a convolutional encoder-decoder neural network.
        contextual spatial information (decoding stage) of an input tensor is merged with
        localization information (encoding stage).
    """

    def __init__(self, out_channels, in_channels=4, depth=5, start_filts=64, up_mode='transpose', merging_mode='concat'):

        """
        args:
            in_channels (int): number of channels in the input tensor. Default is 4 for 'p','b','u','w' dataset
            depth (int): number of MaxPools (stages) in the U-Net
            start_filts (int): number of convolutional filters for the first conv
            up_mode (string): type of upconv operation
                Choices:
                    'transpose': transpose convolution upsampling - parametric upsampling
                    'upsample': nearest neighbour upsampling - non-parametric upsampling
            merging_mode (string): type of merging data from encoder stage(s) to decoder stage(s)
                Choices:
                    'concat': Non-ResNet-style merging of the decoder and encoder activations by concatenating the activations
                    'add': ResNet-style merging of the decoder and encoder activations by adding the activations
                    'final':
        """

        super(UNet, self).__init__()

        self.up_mode = up_mode
        self.merging_mode = merging_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convolutions = []
        self.up_convolutions = []


        # NOTE: up_mode 'upsample' is incompatible with merging_mode 'add'
        if self.up_mode == 'upsample' and self.merging_mode == 'add':
            raise ValueError("up_mode \"upsample\" is not compatible " "with merging_mode \"add\"")

        # creating the encoder path and adding them to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convolutions.append(down_conv)

        # creating the decoder path and adding them to a list
        # NOTE: decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merging_mode=merging_mode)
            self.up_convolutions.append(up_conv)

        self.conv_11 = conv11(outs, self.out_channels)

        # adding the list of modules to the current module
        self.down_convolutions = nn.ModuleList(self.down_convolutions)
        self.up_convolutions = nn.ModuleList(self.up_convolutions)
        self.reset_parameters()

    @staticmethod
    def weight_initializaton(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_parameters(self):

        for i, m in enumerate(self.modules()):
            self.weight_initializaton(m)

    def forward(self, x):

        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convolutions):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i in range(self.depth-1):
            module = self.up_convolutions[i]
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_11(x)

        return x


if __name__ == "__main__":

    """
    TEST
    args: 
        out_channels (int): desired number of output channels that we want the output tensor have  
        depth (int): the desired depth of the UNet
    """

    # creating a pseudo-type input tensor of size batch_size*in_channels*x_dim*y_dim
    batch_size = 16
    in_channels, x_dim, y_dim = 8, 64, 64
    out_channels = 32
    input = np.random.random((batch_size, in_channels, x_dim, y_dim))
    input = Variable(torch.FloatTensor(input))

    # running the UNet model for the given "depth" and desired number of "out_channels" on the input tensor
    model = UNet(out_channels=out_channels, in_channels=in_channels, depth=5, merging_mode='concat')
    model_output = model(input)

    # checking the dimensions of the output and input
    ## output dimension is: batch_size*out_channels*x_dim*y_dim
    print("low resolution input image shape:", input.shape)
    print("high resolution output image shape:", model_output.shape)

    # criterion = nn.MSELoss()
    # loss = torch.sqrt(criterion(input, output[:,:8,:,:]))
    # loss.backward()


