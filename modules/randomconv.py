import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomFixedConv2d(nn.Module):
    """
    A fixed random convolution.
    Wrapper for a more convenient use.
    Basically create a new convolution, initialize it and set
    `requires_grad_(False)` flag.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0,
                 groups=1, dilation=1,
                 bias=True, padding_mode='zeros'):
        super(RandomFixedConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding,
            dilation, groups, bias, padding_mode)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class RandomFixedConvTrans2d(nn.Module):
    """
    A tranposed fixed random convolution.
    Wrapper for a more convenient use.
    Basically create a new convolution, initialize it and set
    `requires_grad_(False)` flag.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,
                 groups=1, dilation=1,
                 bias=True, padding_mode='zeros'):
        super(RandomFixedConvTrans2d, self).__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding,
            groups, bias, dilation, padding_mode)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class RandomFixedSeparableConv2d(nn.Module):
    """
    Depthwise convolution with fixed random kernels followed by
    a trainable 1x1 convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0,
                 dilation=1, bias=True, padding_mode='zeros'):
        super(RandomFixedSeparableConv2d, self).__init__()

        self.fixed_conv = RandomFixedConv2d(
            in_channels, in_channels, kernel_size, stride,
            padding,
            groups=in_channels,
            dilation=dilation, bias=bias, padding_mode=padding_mode)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                            stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.fixed_conv(x)
        x = self.conv1x1(x)
        return x


class RandomFixedSeparableConvTrans2d(nn.Module):
    """
    Transposed depthwise convolution with fixed random kernels preceeded by
    a trainable 1x1 convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0,
                 dilation=1, bias=True, padding_mode='zeros'):
        super(RandomFixedSeparableConvTrans2d, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=bias)

        self.fixed_conv = RandomFixedConvTrans2d(
            out_channels, out_channels, kernel_size, stride,
            padding, output_padding,
            groups=out_channels,
            dilation=dilation, bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.fixed_conv(x)
        return x

