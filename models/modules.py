import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math
import numpy as np
import scipy
import scipy.ndimage

from abc import ABC, abstractmethod


def same_padding(k, s):
    # 2p = - s + k
    x = k - s
    p1 = math.floor(x/2)
    p2 = math.ceil(x/2)
    return p1, p2


def same_padding_transpose(k, s):
    x = k - s
    p = x // 2
    a = x % 2
    return p, a


def calculate_padding(i,o,k,s):
    # 2p = (o-1)*s - i + k
    x = (o-1)*s - i + k
    p1 = math.floor(x/2)
    p2 = math.ceil(x/2)
    return p1, p2, p1, p2


def calculate_padding_transpose(i,o,k,s):
    # 2p = (i-1)*s - o + k + a
    # we use a = 0
    x = (i-1)*s - o + k
    p1 = math.floor(x/2)
    p2 = math.ceil(x/2)
    return p1, p2, p1, p2


class ShortcutConnection(nn.Module):
    """
    For the case inplanes!=planes,
    the shortcut connection should downsample the feature map and
    increase the number of channels.
    This can be done through several options, as described in ResNet paper.

    Option A:
    Adding channels full of 0's.
    Used in ResNet paper for CIFAR10.
    Chosen in the paper because the number of parameters is the same as in plain models.

    Option B:
    1x1 conv + BN.
    Used in ResNet paper for ImageNet models.
    Better performance and the top choice. Better comparison to deep decoder.

    Option C:
    Using 1x1 convs for all shortcut connections instead of just for downsampling.
    worse than A and B.

    Option A code explanation for stride=2:
    every second entry on the horizontal and every second entry on the vertical of
    the feature map are chosen and copied into the output.
    The output is then padded with 0's across the depth dimension so that it is equal to
    `planes`.
    """

    def __init__(self, inplanes, planes, stride=1, option='B'):
        super(ShortcutConnection, self).__init__()

        if inplanes == planes:
            self.shortcut = nn.Identity()
        else:
            assert option in ['A', 'B']
            if option == 'A':
                pad1 = math.ceil((planes - inplanes) / 2)
                pad2 = math.floor((planes - inplanes) / 2)
                self.shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=1, stride=stride),
                    # choosing every stride'th entry
                    nn.ConstantPad3d(padding=(0, 0, 0, 0, pad1, pad2),
                                     value=0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes))

    def forward(self, x):
        x = self.shortcut(x)
        return x


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
        x = self.fixed_conv(x)
        return x

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
        x = self.fixed_conv(x)
        return x

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


class FixedSeparableConv2d(nn.Module):
    """
    SAME padding will be used.
    kernel_size is specified in the fixed_con_params
    """
    def __init__(self, inplanes, planes, stride, fixed_conv_params):
        super(FixedSeparableConv2d, self).__init__()

        self.fixed_conv = self.create_fixed_conv(
            inplanes, stride, fixed_conv_params)
        self.conv1x1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.fixed_conv(x)
        x = self.conv1x1(x)
        return x

    @staticmethod
    def create_fixed_conv(planes, stride, fixed_conv_params):
        conv_type = fixed_conv_params['conv_type']

        if conv_type == 'avgpool':
            if stride == 1:
                fixed_conv = nn.Identity()
            else:
                fixed_conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        elif conv_type == 'maxpool':
            if stride == 1:
                fixed_conv = nn.Identity()
            else:
                fixed_conv = nn.MaxPool2d(kernel_size=stride, stride=stride)

        elif conv_type == 'gaussian':
            kernel_size = fixed_conv_params['kernel_size']
            sigma = fixed_conv_params['sigma']

            fixed_conv = GaussianLayerWPad(planes, kernel_size, sigma, stride)
        else:
            raise RuntimeError("Unknown kernel")
        return fixed_conv


class FixedSeparableConvTranspose2d(nn.Module):
    """
    SAME padding will be used.
    """
    def __init__(self, inplanes, planes, stride, fixed_conv_params):
        super(FixedSeparableConvTranspose2d, self).__init__()

        self.conv1x1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.fixed_conv = self.create_fixed_tranpose_conv(
            planes, stride, fixed_conv_params)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.fixed_conv(x)
        return x

    @staticmethod
    def create_fixed_tranpose_conv(planes, stride, fixed_conv_params):
        conv_type = fixed_conv_params['conv_type']

        if stride != 1:
            raise RuntimeError("Use bilinear upsamling")
        else:
            if conv_type == 'avgpool' or conv_type == 'maxpool':
                fixed_conv = nn.Identity()

            elif conv_type == 'gaussian':
                kernel_size = fixed_conv_params['kernel_size']
                sigma = fixed_conv_params['sigma']

                # for stride==1 gaussian layer and
                # transposed gaussian layer are the same
                fixed_conv = GaussianLayerWPad(planes, kernel_size, sigma,
                                               stride=1)
                # the transposed option
                # fixed_conv = GaussianLayerTransposeWPad(planes, kernel_size,
                #                                         sigma, stride)
            else:
                raise RuntimeError("Unknown kernel")
        return fixed_conv


class FixedBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, fixed_conv_params, option='B'):
        super(FixedBlock, self).__init__()

        self.fixed_sep_conv1 = FixedSeparableConv2d(
            inplanes, planes, stride, fixed_conv_params)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.fixed_sep_conv2 = FixedSeparableConv2d(
            planes, planes, 1, fixed_conv_params)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.shortcut = ShortcutConnection(inplanes, planes, stride, option)

    def forward(self, x):
        out = self.relu1(self.bn1(self.fixed_sep_conv1(x)))
        out = self.bn2(self.fixed_sep_conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.shortcut = ShortcutConnection(inplanes, planes, stride, option)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, option='B'):
        super(BottleneckBlock, self).__init__()

        bottleneck_planes = planes // self.expansion

        self.conv1 = nn.Conv2d(inplanes, bottleneck_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(bottleneck_planes, bottleneck_planes,
                               kernel_size=3,
                               stride=stride, padding=1,
                               bias=False)  # downsample with 3x3 conv
        self.bn2 = nn.BatchNorm2d(bottleneck_planes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(bottleneck_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU()

        self.shortcut = ShortcutConnection(inplanes, planes, stride, option)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet_CIFAR(nn.Module):

    def __init__(self, block, n_blocks, num_classes=10, option='B'):
        super(ResNet_CIFAR, self).__init__()

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()
        self.stage1 = self._make_stage(block, n_blocks[0], 16, 16, stride=1,
                                       option=option)
        self.stage2 = self._make_stage(block, n_blocks[1], 16, 32, stride=2,
                                       option=option)
        self.stage3 = self._make_stage(block, n_blocks[2], 32, 64, stride=2,
                                       option=option)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, block, n_blocks, inplanes, planes, stride, option):
        stage = nn.Sequential()
        stage.add_module('block0', block(inplanes, planes, stride, option))
        for i in range(1, n_blocks):
            stage.add_module('block' + str(i), block(planes, planes, 1, option))
        return stage

    def forward(self, x):
        x = self.relu0(self.bn0(self.conv0(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)  # global average pooling
        x = self.flatten(x)
        x = self.fc(x)
        return x


class FixedResNet_CIFAR(nn.Module):
    def __init__(self, n_blocks, fixed_conv_params, num_classes=10,
                 k=1, option='B', fully_fixed=False):
        """Creates a ResNet for cifar10 where each normal convolution
        is replaced by a fixed convolutional filter followed by a 1x1 conv.

        Args:
            n_blocks: list with the number of blocks for each stage
            fixed_conv_params: dict with parameters needed to create
                a fixed convolutional filter
            num_classes: number of output classes
            k: widening factor, as used in WideResNets
            option: the type of shortcut connection to use.
                Refer to ShortcutConnection
            fully_fixed: Specifies if "stage0" of the net should use
                a learnable convolution or a fixed one.
        """
        super(FixedResNet_CIFAR, self).__init__()

        if fully_fixed:
            self.stage0 = nn.Sequential(
                FixedSeparableConv2d(3, 16*k, stride=1,
                                     fixed_conv_params=fixed_conv_params),
                nn.BatchNorm2d(16*k),
                nn.ReLU())
        else:
            self.stage0 = nn.Sequential(
                nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16*k),
                nn.ReLU())

        self.stage1 = self._make_stage(n_blocks[0], 16*k, 16*k, 1,
                                       fixed_conv_params, option)
        self.stage2 = self._make_stage(n_blocks[1], 16*k, 32*k, 2,
                                       fixed_conv_params, option)
        self.stage3 = self._make_stage(n_blocks[2], 32*k, 64*k, 2,
                                       fixed_conv_params, option)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64*k, num_classes)

    def _make_stage(self, n_blocks, inplanes, planes, stride,
                    fixed_conv_params, option):
        stage = nn.Sequential()
        block = FixedBlock(inplanes, planes, stride,
                           fixed_conv_params, option)
        stage.add_module('block0', block)
        for i in range(1, n_blocks):
            block = FixedBlock(planes, planes, 1,
                               fixed_conv_params, option)
            stage.add_module('block' + str(i), block)
        return stage

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)  # global average pooling
        x = self.flatten(x)
        x = self.fc(x)
        return x


# for s=1 equal to GaussianLayerWPad, for s=2 bilinear should be used
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
        self.conv = GaussianLayerTranspose(planes, kernel_size, sigma, stride,
                                           padding=kernel_size-1,
                                           output_padding=output_padding)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x

