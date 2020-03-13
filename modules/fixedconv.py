import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import scipy
import scipy.ndimage

from modules.functions import same_padding

from abc import ABC, abstractmethod


def create_fixed_conv(planes, stride, fixed_conv_params):
    conv_type = fixed_conv_params['conv_type']

    if conv_type == 'bilinear':
        if stride != 1:
            fixed_conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            fixed_conv = None

    elif conv_type == 'gaussian':
        kernel_size = fixed_conv_params['kernel_size']
        sigma = fixed_conv_params['sigma']

        fixed_conv = GaussianLayerWPad(planes, kernel_size, sigma, stride)
    else:
        raise RuntimeError("Unknown kernel")
    return fixed_conv


def create_fixed_conv_tranpose(planes, stride, fixed_conv_params):
    conv_type = fixed_conv_params['conv_type']

    if conv_type == 'bilinear':
        if stride != 1:
            fixed_conv = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            fixed_conv = None  # Identity

    elif conv_type == 'gaussian':
        if stride != 1:
            raise RuntimeError(
                "Using transposed convolutions with strides generally works bad."
                "Consider bilinear upsamling + transposed conv with stride 1. "
                "If you are sure you want to use this code anyway, remove this line.")
        kernel_size = fixed_conv_params['kernel_size']
        sigma = fixed_conv_params['sigma']

        fixed_conv = GaussianLayerTransposeWPad(planes, kernel_size, sigma, stride)
    else:
        raise RuntimeError("Unknown kernel")
    return fixed_conv


def get_fixed_conv_params(conv_type, bilin_interpol=False, n=None, sigma=None, amount=None):
    """

    Args:
        conv_type (str): first letter specifies the type of filter to use.
            G - Gaussian
            B - Bilinear interpolation
            S - Sharpening
            R - Random
        bilin_interpol (bool): if True, interpolate with bilinear
            interpolation and then use conv with stride 1,
            if False, use convs with stride!=1 for upsampling and downsampling.
        n (int): kernel_size for gaussian and sharpening filters
        sigma (float): standard deviation, sigma value for gaussian and
            sharpening filters kernel
        amount (float): amount of sharpening to apply.
            Check https://en.wikipedia.org/wiki/Unsharp_masking for details


    Returns:
        dict: dictionary with parameters for the fixed convolution
    """
    if conv_type == 'G':
        if n is None: n = 3
        if sigma is None: sigma = 0.8
        fixed_conv_params = {'conv_type': 'gaussian',
                             'bilin_interpol': bilin_interpol,
                             'kernel_size': n, 'sigma': sigma}

    elif conv_type == 'B':
        fixed_conv_params = {'conv_type': 'bilinear',
                             'bilin_interpol': False}
        # flag is set to false, because bilinear interpolation will be
        # done explicitly

    else:
        raise RuntimeError("conv_type not defined")
    return fixed_conv_params


class FixedConv2d(nn.Module, ABC):
    """
    1. Depthwise separable filter with a fixed conv kernel
    2. Isn't tracked by autograd
    3. Downsamples for stride > 1
    4. inplanes=planes=groups
    """
    def __init__(self, planes, kernel_size, stride=1, padding=0, *args):
        super(FixedConv2d, self).__init__()

        self.fixed_conv = nn.Conv2d(planes, planes, groups=planes,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=False)
        fixed_kernel = self.create_kernel(kernel_size, *args)
        self.fixed_conv.weight.data.copy_(fixed_kernel)  # broadcasts 2d to 4d
        self.requires_grad_(False)

    def forward(self, x):
        return self.fixed_conv(x)

    @abstractmethod
    def create_kernel(self, kernel_size, *args):
        """ should return a 2d, 3d or 4d torch.Tensor """
        pass


class FixedConvTranspose2d(nn.Module, ABC):
    """
    1. Depthwise separable filter with a fixed conv transposed kernel
    2. Isn't tracked by autograd
    3. Upsamples for stride > 1
    4. inplanes=planes=groups
    """
    def __init__(self, planes, kernel_size, stride=1, padding=0, output_padding=0, *args):
        super(FixedConvTranspose2d, self).__init__()

        self.fixed_conv = nn.ConvTranspose2d(
            planes, planes, groups=planes, kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding, bias=False)
        fixed_kernel = self.create_kernel(kernel_size, *args)
        self.fixed_conv.weight.data.copy_(fixed_kernel)  # broadcasts 2d to 4d
        self.requires_grad_(False)

    def forward(self, x):
        return self.fixed_conv(x)

    @abstractmethod
    def create_kernel(self, kernel_size, *args):
        """ should return a 2d, 3d or 4d torch.Tensor """
        pass


class GaussianLayer(FixedConv2d):
    """
    Performs gaussian blur with kernel_size=n and std=sigma
    Best to use reflection padding before the GaussianLayer.

    Good values:
    sigma=0.8 for n=3
    sigma=0.75 for n=4
    """

    def __init__(self, planes, kernel_size, sigma, stride=1, padding=0):
        super(GaussianLayer, self).__init__(planes, kernel_size, stride,
                                            padding, sigma)

    def create_kernel(self, kernel_size, sigma):
        n = kernel_size
        mat = np.zeros((n, n))
        if n % 2 == 1:  # odd
            mat[n // 2, n // 2] = 1
        else:  # even
            mat[n//2-1:n//2+1, n//2-1:n//2+1] = 0.25
        k = scipy.ndimage.gaussian_filter(mat, sigma)
        return torch.from_numpy(k)


class GaussianLayerTranspose(FixedConvTranspose2d):
    """
    Performs gaussian blur with kernel_size=n and std=sigma
    Best to use reflection padding before the GaussianLayerTranspose.

    Good values:
    sigma=0.8 for n=3
    sigma=0.75 for n=4
    """

    def __init__(self, planes, kernel_size, sigma, stride=1, padding=0, output_padding=0):
        super(GaussianLayerTranspose, self).__init__(planes, kernel_size, stride,
                                                     padding, output_padding, sigma)

    def create_kernel(self, kernel_size, sigma):
        n = kernel_size
        mat = np.zeros((n, n))
        if n % 2 == 1:  # odd
            mat[n // 2, n // 2] = 1
        else:  # even
            mat[n//2-1:n//2+1, n//2-1:n//2+1] = 0.25
        k = scipy.ndimage.gaussian_filter(mat, sigma)
        return torch.from_numpy(k)


class GaussianLayerWPad(nn.Module):
    """
    "SAME" Reflection Padding + GaussianLayer
    We create Same padding with ReflectionPad2d and then use p=0 in conv.
    """
    def __init__(self, planes, kernel_size, sigma, stride=1):
        super(GaussianLayerWPad, self).__init__()
        p1, p2 = same_padding(kernel_size, stride)
        self.pad = nn.ReflectionPad2d((p1, p2, p1, p2))
        self.conv = GaussianLayer(planes, kernel_size, sigma, stride,
                                  padding=0)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class GaussianLayerTransposeWPad(nn.Module):
    """
    "SAME" Reflection Padding + GaussianLayerTranspose

    We create Same padding with ReflectionPad2d.
     Then we have to compensate for that padding and use FULL padding in
     transposed convolution.

    This assymetry only happens, because we have padded separately before the
    trans conv.
    In normal circumstances, a simple half padding with p=k//2 would be used.

    Note:
        It is best to not use this with stride > 1. Checkboard artfacts and
        undefined output shape (output_padding should be used).
    """
    def __init__(self, planes, kernel_size, sigma, stride=1):
        #assert stride == 1, "Use bilinear upsampling or other upsampling technique"
        super(GaussianLayerTransposeWPad, self).__init__()
        p1, p2 = same_padding(kernel_size, stride)
        output_padding = 1 if (kernel_size-stride) % 2 == 1 else 0
        self.pad = nn.ReflectionPad2d((p1, p2, p1, p2))
        if stride == 1:
            """ There is a problem with the GaussianLayerTranpose not working 
            for a combination of values k=4,s=1,o=1 
            Regular GaussianLayer is equivalent for s=1 and can be used here"""
            self.conv = GaussianLayer(
                planes, kernel_size, sigma, stride, padding=0)
        else:  # for stride > 1 it works fine
            self.conv = GaussianLayerTranspose(
                planes, kernel_size, sigma, stride,
                padding=kernel_size-1, output_padding=output_padding)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class FixedSeparableConv2d(nn.Module):
    """
    SAME padding will be used.
    kernel_size is specified in the fixed_con_params
    """
    def __init__(self, inplanes, planes, stride, fixed_conv_params):
        super(FixedSeparableConv2d, self).__init__()

        # bilinear interpolation + fixedconv
        if stride > 1 and (fixed_conv_params['bilin_interpol'] is True):
            inter = nn.AvgPool2d(stride, stride)
            fixed_conv = create_fixed_conv(inplanes, 1, fixed_conv_params)
        # fixedconv
        else:
            inter = None
            fixed_conv = create_fixed_conv(inplanes, stride, fixed_conv_params)

        conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                            stride=1, padding=0, bias=False)

        layers = filter(lambda x: x is not None, [inter, fixed_conv, conv1x1])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FixedSeparableConvTranspose2d(nn.Module):
    """
    SAME padding will be used.
    """
    def __init__(self, inplanes, planes, stride, fixed_conv_params):
        super(FixedSeparableConvTranspose2d, self).__init__()

        conv1x1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

        # bilinear interpolation + fixedconv
        if stride > 1 and (fixed_conv_params['bilin_interpol'] is True):
            inter = nn.UpsamplingBilinear2d(scale_factor=stride)
            fixed_conv = create_fixed_conv_tranpose(planes, 1, fixed_conv_params)
        # fixedconv
        else:
            inter = None
            fixed_conv = create_fixed_conv_tranpose(planes, stride, fixed_conv_params)

        layers = filter(lambda x: x is not None, [conv1x1, inter, fixed_conv])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
