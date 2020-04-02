from modules.resnet import *
from modules.preact_resnet import *
from functools import partial


FixedBottleneckBlock1 = partial(FixedBottleneckBlock, factor=1)
FixedBottleneckBlock2 = partial(FixedBottleneckBlock, factor=4)
FixedBottleneckBlock3 = partial(FixedBottleneckBlock, factor=0.25)


def resnet20(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [3, 3, 3],
                  num_classes, k, fixed, fully_fixed)


def resnet32(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [5, 5, 5],
                  num_classes, k, fixed, fully_fixed)


def resnet44(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [7, 7, 7],
                  num_classes, k, fixed, fully_fixed)


def resnet56(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [9, 9, 9],
                  num_classes, k, fixed, fully_fixed)


def resnet110(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [18, 18, 18],
                  num_classes, k, fixed, fully_fixed)


def resnet1202(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BasicBlock, [200, 200, 200],
                  num_classes, k, fixed, fully_fixed)


def resnet164(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BottleneckBlock, [18, 18, 18],
                  num_classes, k, fixed, fully_fixed)


def resnet1001(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return ResNet(BottleneckBlock, [111, 111, 111],
                  num_classes, k, fixed, fully_fixed)


def preact_resnet20(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [3, 3, 3],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet32(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [5, 5, 5],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet44(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [7, 7, 7],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet56(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [9, 9, 9],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet110(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [18, 18, 18],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet1202(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBasicBlock, [200, 200, 200],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet164(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBottleneckBlock, [18, 18, 18],
                        num_classes, k, fixed, fully_fixed)

def preact_resnet164_fixedbottleneck(num_classes=10, k=1, fixed=True,
                                    fully_fixed=False):
    assert fixed, 'This architecture is a fixed one, use fixed=True!'
    return PreActResNet(FixedBottleneckBlock, [18, 18, 18],
                        num_classes, k, fixed, fully_fixed)

def preact_resnet1001(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBottleneckBlock, [111, 111, 111],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet29(num_classes=10, k=1, fixed=False, fully_fixed=False):
    return PreActResNet(PreActBottleneckBlock, [3, 3, 3],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet29_fixedbottleneck1(num_classes=10, k=1, fixed=True,
                                     fully_fixed=False):
    assert fixed, 'This architecture is a fixed one, use fixed=True!'
    return PreActResNet(FixedBottleneckBlock1, [3, 3, 3],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet29_fixedbottleneck2(num_classes=10, k=1, fixed=True,
                                     fully_fixed=False):
    assert fixed, 'This architecture is a fixed one, use fixed=True!'
    return PreActResNet(FixedBottleneckBlock2, [3, 3, 3],
                        num_classes, k, fixed, fully_fixed)


def preact_resnet29_fixedbottleneck3(num_classes=10, k=1, fixed=True,
                                     fully_fixed=False):
    assert fixed, 'This architecture is a fixed one, use fixed=True!'
    return PreActResNet(FixedBottleneckBlock3, [3, 3, 3],
                        num_classes, k, fixed, fully_fixed)

