import torch
import torch.nn as nn
import torch.nn.functional as F


def conv33(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):

    """
        3x3 2d convolution parts with, kernel_size of 3, stride of 1, and unity padding
    """

    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)

def upconv22(in_channels, out_channels, mode='transpose'):

    """
        up-conv operation in the decoding path
        we can consider either ConvTranspose2d (:parametric) or Upsample operations (:non-parametric)
    """

    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        # mode in {'nearest','linear','bilinear','bicubic','trilinear}
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv11(in_channels, out_channels))

def conv11(in_channels, out_channels, groups=1):

    """
        1x1 2d convolution parts with, kernel_size of 1, and stride of 1
    """

    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):

    """
        Down convolution block that performs 2 stages of 3x3 convolutions and 1 MaxPooling
        with a ReLU activation following each 3x3 convolution operation
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv33(self.in_channels, self.out_channels)
        self.conv2 = conv33(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pre_pooling = x
        if self.pooling:
            x = self.pool(x)

        return x, pre_pooling


class UpConv(nn.Module):

    """
        Up convolution block that performs 2 stages of 3x3 convolutions and 1 UpConvolution
        with a ReLU activation following each 3x3 convolution operation
    """

    def __init__(self, in_channels, out_channels, merging_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merging_mode = merging_mode
        self.up_mode = up_mode
        self.upconv = upconv22(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merging_mode == 'concat':
            self.conv1 = conv33(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv33(self.out_channels, self.out_channels)

        self.conv2 = conv33(self.out_channels, self.out_channels)

    def forward(self, from_encoder, from_decoder):

        """
        Forward pass:
        args:
            from_encoder: tensor that gets added during UpConv from the encoder pathway
            from_decoder: upconvolutioned tensor from the decoder pathway
        """

        from_decoder = self.upconv(from_decoder)

        if self.merging_mode == 'concat':
            x = torch.cat((from_decoder, from_encoder), 1)
        else:
            x = from_decoder + from_encoder

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x
