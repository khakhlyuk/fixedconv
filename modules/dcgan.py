import torch
import torch.nn as nn
from modules.randomconv import *


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, k=1, fixed=False, fixed_option='A'):
        super(Generator, self).__init__()

        assert fixed_option in ['A', 'B', 'C']

        if not fixed:
            # all convs are trainable
            n_channels = [ngf * 8 * k, ngf * 4 * k, ngf * 2 * k, ngf * 1 * k]
            conv_module1 = nn.ConvTranspose2d
            conv_module2 = nn.ConvTranspose2d
            conv_module3 = nn.ConvTranspose2d
            conv_module4 = nn.ConvTranspose2d
            conv_module5 = nn.ConvTranspose2d
        elif fixed and fixed_option == 'A':
            # all convs are fixed
            n_channels = [ngf * 8 * k, ngf * 4 * k, ngf * 2 * k, ngf * 1 * k]
            conv_module1 = RandomFixedSeparableConvTrans2d
            conv_module2 = RandomFixedSeparableConvTrans2d
            conv_module3 = RandomFixedSeparableConvTrans2d
            conv_module4 = RandomFixedSeparableConvTrans2d
            conv_module5 = RandomFixedSeparableConvTrans2d
        elif fixed and fixed_option == 'B':
            # first conv is trainable, others are fixed
            n_channels = [ngf * 8 * 1, ngf * 4 * k, ngf * 2 * k, ngf * 1 * k]
            conv_module1 = nn.ConvTranspose2d
            conv_module2 = RandomFixedSeparableConvTrans2d
            conv_module3 = RandomFixedSeparableConvTrans2d
            conv_module4 = RandomFixedSeparableConvTrans2d
            conv_module5 = RandomFixedSeparableConvTrans2d
        elif fixed and fixed_option == 'C':
            # first and last convs is trainable, others are fixed
            n_channels = [ngf * 8 * 1, ngf * 4 * k, ngf * 2 * k, ngf * 1 * 1]
            conv_module1 = nn.ConvTranspose2d
            conv_module2 = RandomFixedSeparableConvTrans2d
            conv_module3 = RandomFixedSeparableConvTrans2d
            conv_module4 = RandomFixedSeparableConvTrans2d
            conv_module5 = nn.ConvTranspose2d

        # input Z. nz x 1 x 1

        self.conv1 = conv_module1(nz, n_channels[0], 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.relu1 = nn.ReLU()

        # state size. (ngf*8*k) x 4 x 4

        self.conv2 = conv_module2(n_channels[0], n_channels[1], 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels[1])
        self.relu2 = nn.ReLU()

        # state size. (ngf*4*k) x 8 x 8

        self.conv3 = conv_module3(n_channels[1], n_channels[2], 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_channels[2])
        self.relu3 = nn.ReLU()

        # state size. (ngf*2*k) x 16 x 16

        self.conv4 = conv_module4(n_channels[2], n_channels[3], 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(n_channels[3])
        self.relu4 = nn.ReLU()

        # state size. (ngf*1*k) x 32 x 32

        self.conv5 = conv_module5(n_channels[3], nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

        # state size. (nc) x 64 x 64

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, k=1, fixed=False, fixed_option='A'):
        super(Discriminator, self).__init__()

        assert fixed_option in ['A', 'B', 'C']

        if not fixed:
            # all convs are trainable
            n_channels = [ndf * 1 * k, ndf * 2 * k, ndf * 4 * k, ndf * 8 * k]
            conv_module1 = nn.Conv2d
            conv_module2 = nn.Conv2d
            conv_module3 = nn.Conv2d
            conv_module4 = nn.Conv2d
            conv_module5 = nn.Conv2d
        elif fixed and fixed_option == 'A':
            # all convs are fixed
            n_channels = [ndf * 1 * k, ndf * 2 * k, ndf * 4 * k, ndf * 8 * k]
            conv_module1 = RandomFixedSeparableConv2d
            conv_module2 = RandomFixedSeparableConv2d
            conv_module3 = RandomFixedSeparableConv2d
            conv_module4 = RandomFixedSeparableConv2d
            conv_module5 = RandomFixedSeparableConv2d
        elif fixed and fixed_option == 'B':
            # last conv is trainable, others are fixed
            n_channels = [ndf * 1 * k, ndf * 2 * k, ndf * 4 * k, ndf * 8 * 1]
            conv_module1 = RandomFixedSeparableConv2d
            conv_module2 = RandomFixedSeparableConv2d
            conv_module3 = RandomFixedSeparableConv2d
            conv_module4 = RandomFixedSeparableConv2d
            conv_module5 = nn.Conv2d
        elif fixed and fixed_option == 'C':
            # first and last convs is trainable, others are fixed
            n_channels = [ndf * 1 * 1, ndf * 2 * k, ndf * 4 * k, ndf * 8 * 1]
            conv_module1 = nn.Conv2d
            conv_module2 = RandomFixedSeparableConv2d
            conv_module3 = RandomFixedSeparableConv2d
            conv_module4 = RandomFixedSeparableConv2d
            conv_module5 = nn.Conv2d

        # x is (nc) x 64 x 64

        self.conv1 = conv_module1(nc, n_channels[0], 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.lrelu1 = nn.LeakyReLU(0.2)

        # state size. (ndf*1*k) x 32 x 32

        self.conv2 = conv_module2(n_channels[0], n_channels[1], 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels[1])
        self.lrelu2 = nn.LeakyReLU(0.2)

        # state size. (ndf*2*k) x 16 x 16

        self.conv3 = conv_module3(n_channels[1], n_channels[2], 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_channels[2])
        self.lrelu3 = nn.LeakyReLU(0.2)

        # state size. (ndf*4*k) x 8 x 8

        self.conv4 = conv_module4(n_channels[2], n_channels[3], 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(n_channels[3])
        self.lrelu4 = nn.LeakyReLU(0.2)

        # state size. (ndf*8*k) x 4 x 4

        self.conv5 = conv_module5(n_channels[3], 1, 4, 1, 0, bias=False)
        self.sigm = nn.Sigmoid()

        # state size. 1 x 1 x 1

    def forward(self, x):
        x = self.lrelu1(self.bn1(self.conv1(x)))
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.sigm(self.conv5(x))
        return x
