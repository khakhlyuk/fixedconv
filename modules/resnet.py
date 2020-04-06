import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from modules.fixedconv import *


class ShortcutConnection(nn.Module):
    """
    For the case in_channels!=out_channels,
    the shortcut connection should downsample the feature map and
    increase the number of channels.
    This can be done through several options, as described in ResNet paper.

    Option A:
    Adding channels full of 0's.
    Used in ResNet paper for CIFAR10.
    Chosen in the paper because the number of parameters is the same as in plain modules.

    Option B:
    1x1 conv + BN.
    Used in ResNet paper for ImageNet modules.
    Better performance and the top choice. Better comparison to deep decoder.

    Option C:
    Using 1x1 convs for all shortcut connections instead of just for downsampling.
    worse than A and B.

    Option A code explanation for stride=2:
    every second entry on the horizontal and every second entry on the vertical of
    the feature map are chosen and copied into the output.
    The output is then padded with 0's across the depth dimension so that it is equal to
    `out_channels`.
    """

    def __init__(self, in_channels, out_channels, stride=1, option='B'):
        super(ShortcutConnection, self).__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            assert option in ['A', 'B']
            if option == 'A':
                pad1 = math.ceil((out_channels - in_channels) / 2)
                pad2 = math.floor((out_channels - in_channels) / 2)
                self.shortcut = nn.Sequential(
                    nn.MaxPool2d(kernel_size=1, stride=stride),
                    # choosing every stride'th entry
                    nn.ConstantPad3d(padding=(0, 0, 0, 0, pad1, pad2),
                                     value=0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.shortcut(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, fixed=False):
        super(BasicBlock, self).__init__()

        # choose fixed or trainable
        if fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self.conv1 = conv_module(in_channels, out_channels, kernel_size=3,
                                 stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv_module(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut = ShortcutConnection(in_channels, out_channels, stride)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, fixed=False):
        super(BottleneckBlock, self).__init__()
        bottleneck_channels = out_channels // self.expansion

        # choose fixed or trainable
        if fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = conv_module(bottleneck_channels, bottleneck_channels,
                                 kernel_size=3,
                                 stride=stride, padding=1,
                                 bias=False)  # downsample with 3x3 conv
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.shortcut = ShortcutConnection(in_channels, out_channels, stride)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-10 and CIFAR-100.
     The `fixed` option allows to replace trainable convolutions
     with fixed separable convolutions.

    Args:
        block_module: type of block to use BasicBlock or BottleneckBlock
        n_blocks: list with the number of blocks for each stage
        num_classes: number of output classes
        k: widening factor, as used in WideResNets
        fixed: trainable or fixed ResNet
        fully_fixed: Specifies if "stage0" of the network should use
            a learnable convolution or a fixed one.
    """

    def __init__(self, block_module, n_blocks, num_classes=10, k=1,
                 fixed=False, fully_fixed=False):
        super(ResNet, self).__init__()

        # number of channels in each stage
        base_channels = 16
        n_channels = [base_channels * 1 * k,
                      base_channels * 1 * k * block_module.expansion,
                      base_channels * 2 * k * block_module.expansion,
                      base_channels * 4 * k * block_module.expansion]
        n_channels = [int(x) for x in n_channels]  # for float k

        # we only fix the first conv in fully_fixed network
        if fixed and fully_fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self.conv = conv_module(3, n_channels[0], kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[0])
        self.relu = nn.ReLU()
        self.stage1 = self._make_stage(block_module, n_blocks[0],
                                       n_channels[0], n_channels[1], stride=1,
                                       fixed=fixed)
        self.stage2 = self._make_stage(block_module, n_blocks[1],
                                       n_channels[1], n_channels[2], stride=2,
                                       fixed=fixed)
        self.stage3 = self._make_stage(block_module, n_blocks[2],
                                       n_channels[2], n_channels[3], stride=2,
                                       fixed=fixed)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_channels[3], num_classes)

    def _make_stage(self, block_module, n_blocks, in_channels, out_channels,
                    stride, fixed):
        stage = nn.Sequential()
        stage.add_module('block0',
                         block_module(in_channels, out_channels,
                                      stride, fixed))
        for i in range(1, n_blocks):
            stage.add_module('block' + str(i),
                             block_module(out_channels, out_channels,
                                          1, fixed))
        return stage

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)  # global average pooling
        x = self.flatten(x)
        x = self.fc(x)
        return x

