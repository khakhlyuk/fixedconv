import torch
import torch.nn as nn

from modules.fixedconv import RandomFixedSeparableConv2d


class ShortcutConnection(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShortcutConnection, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.shortcut(x)
        return x


class SeparableConv2d(nn.Module):
    """
    Depthwise convolution with fixed random kernels followed by
    a trainable 1x1 convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0,
                 dilation=1, bias=True, padding_mode='zeros'):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride,
            padding, groups=in_channels,
            dilation=dilation, bias=bias, padding_mode=padding_mode)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EntryFlow(nn.Module):
    '''
    Original Outputs of each layer

    Input entry flow: 299 x 299 x 3
    ---
    150 x 150 x 32
    150 x 150 x 64
    ---
    150 x 150 x 128
    150 x 150 x 128
    75 x 75 x 128
    ---
    75 x 75 x 256
    75 x 75 x 256
    38 x 38 x 256
    ---
    38 x 38 x 728
    38 x 38 x 728
    19 x 19 x 728
    ---
    Output entry flow: 19 x 19 x 728

    ---------------------------------
    Our Outputs of each layer

    Input entry flow: 64 x 64 x 3
    ---
    64 x 64 x 128
    64 x 64 x 128
    ---
    64 x 64 x 256
    64 x 64 x 256
    32 x 32 x 256
    ---
    32 x 32 x 728
    32 x 32 x 728
    16 x 16 x 728
    ---
    Output entry flow: 16 x 16 x 728
    '''

    def __init__(self, conv_module, size):
        super(EntryFlow, self).__init__()

        if size == 32:
            # 32x32x3 to 64x64x256 to 32x32x256
            self.conv1 = nn.Sequential(
                # 32x32x3x512 vs 150x150x3x32
                nn.Conv2d(3, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

            # 32x32x256 to 32x32x768 to 16x16x728
            self.conv2_residual = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_module(256, 728, 3, padding=1),
                nn.BatchNorm2d(728),
                nn.ReLU(inplace=True),
                conv_module(728, 728, 3, padding=1),
                nn.BatchNorm2d(728),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

            self.conv2_shortcut = ShortcutConnection(256, 728, 2)
            self.conv3_residual = nn.Identity()
            self.conv3_shortcut = nn.Identity()

        elif size == 64:
            # 64x64x3 to 64x64x128 to 64x64x128
            self.conv1 = nn.Sequential(
                # 64x64x3x128 vs 150x150x3x32
                nn.Conv2d(3, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            # 64x64x128 to 64x64x256 to 32x32x256
            self.conv2_residual = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_module(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                conv_module(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

            self.conv2_shortcut = ShortcutConnection(128, 256, 2)

            # 32x32x256 to 32x32x768 to 16x16x728
            self.conv3_residual = nn.Sequential(
                nn.ReLU(inplace=True),
                conv_module(256, 728, 3, padding=1),
                nn.BatchNorm2d(728),
                nn.ReLU(inplace=True),
                conv_module(728, 728, 3, padding=1),
                nn.BatchNorm2d(728),
                nn.MaxPool2d(3, stride=2, padding=1),
            )

            self.conv3_shortcut = ShortcutConnection(256, 728, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self, conv_module):
        super(MiddleFLowBlock, self).__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_module(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_module(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            conv_module(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)
        return shortcut + residual


class MiddleFlow(nn.Module):
    def __init__(self, conv_module, length):
        super(MiddleFlow, self).__init__()
        # """then through the middle flow which is repeated eight times"""
        self.middle_block = self._make_flow(conv_module, length)

    def forward(self, x):
        x = self.middle_block(x)
        return x

    def _make_flow(self, conv_module, length):
        blocks = []
        for i in range(length):
            blocks.append(MiddleFLowBlock(conv_module))

        return nn.Sequential(*blocks)


class ExitFLow(nn.Module):

    def __init__(self, conv_module):
        super(ExitFLow, self).__init__()
        # 16x16x728
        self.residual = nn.Sequential(
            nn.ReLU(),
            conv_module(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            conv_module(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = ShortcutConnection(728, 1024, 2)

        # 8x8x1024
        self.conv = nn.Sequential(
            conv_module(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            conv_module(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        # 8x8x2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 1x1x2048

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)

        return output


class Xception(nn.Module):

    def __init__(self, num_classes=100, fixed=False,
                 middle_flow_length=8, input_size=32):
        super(Xception, self).__init__()


        if fixed:
            conv_module = RandomFixedSeparableConv2d
        else:
            conv_module = SeparableConv2d

        # 64x64x3
        self.entry_flow = EntryFlow(conv_module, input_size)
        # 16x16x728
        self.middle_flow = MiddleFlow(conv_module, middle_flow_length)
        # 16x16x728
        self.exit_flow = ExitFLow(conv_module)
        # 1x1x2048
        self.fc = nn.Linear(2048, num_classes)
        # 1x1x2048

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


