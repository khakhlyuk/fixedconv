import torch
import torch.nn as nnя
import torch.nn.functional as F
import torchvision
import math

from modules.randomconv import *


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 fixed=False,
                 preact=False):
        super(PreActBasicBlock, self).__init__()

        # choose fixed or trainable
        if fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self._preact = preact

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv_module(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_module(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1,bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('conv', nn.Conv2d(
                in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False))

    def forward(self, x):
        if self._preact:
            # preactivation for residual AND shortcut
            # done after applying 1x1 conv shortcut, not done after Identity.
            x = F.relu(self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        y += self.shortcut(x)
        return y


class PreActBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 fixed=False,
                 preact=False):
        super(PreActBottleneckBlock, self).__init__()

        # choose fixed or trainable
        if fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self._preact = preact

        bottleneck_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = conv_module(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,  # downsample with 3x3 conv
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.shortcut = nn.Sequential()  # identity
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))

    def forward(self, x):
        if self._preact:
            x = F.relu(self.bn1(x), inplace=True)  # shortcut after preactivation
            y = self.conv1(x)
        else:
            # preactivation only for residual path
            y = F.relu(self.bn1(x), inplace=True)
            y = self.conv1(y)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.conv3(y)

        y += self.shortcut(x)
        return y

class PreActResNet(nn.Module):
    def __init__(self, block_module, n_blocks, num_classes=10, k=1,
                 fixed=False, fully_fixed=False):
        super(PreActResNet, self).__init__()

        # number of channels in each stage
        n_channels = [16, 32, 64]
        # adding widening factor k
        n_channels = [x * k for x in n_channels]

        # depth = 110
        # assert block_type in ['basic', 'bottleneck']
        # if block_type == 'basic':
        #     block = BasicBlock
        #     n_blocks_per_stage = (depth - 2) // 6
        #     assert n_blocks_per_stage * 6 + 2 == depth
        # else:
        #     block = BottleneckBlock
        #     n_blocks_per_stage = (depth - 2) // 9
        #     assert n_blocks_per_stage * 9 + 2 == depth

        if fixed and fully_fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = nn.Conv2d

        self.conv = conv_module(3, n_channels[0], kernel_size=3, stride=1,
                                padding=1, bias=False)

        self.stage1 = self._make_stage(block_module, n_blocks[0],
                                       n_channels[0], n_channels[0], stride=1,
                                       fixed=fixed)
        self.stage2 = self._make_stage(block_module, n_blocks[1],
                                       n_channels[0], n_channels[1], stride=2,
                                       fixed=fixed)
        self.stage3 = self._make_stage(block_module, n_blocks[2],
                                       n_channels[1], n_channels[2], stride=2,
                                       fixed=fixed)
        self.bn = nn.BatchNorm2d(n_channels[2])
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(n_channels[2], num_classes)

    def _make_stage(self, block_module, n_blocks, in_channels, out_channels,
                    stride, fixed):
        stage = nn.Sequential()

        stage.add_module('block0',
                         block_module(in_channels, out_channels,
                                      stride, fixed, preact=True))
        for i in range(1, n_blocks):
            stage.add_module('block' + str(i),
                             block_module(out_channels, out_channels,
                                          1, fixed, preact=False))
        return stage

    def forward(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.relu(self.bn(x))  # apply BN and ReLU before average pooling
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
