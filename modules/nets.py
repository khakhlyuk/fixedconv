from modules.resnet import *


def resnet20():
    return ResNet_CIFAR(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet_CIFAR(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet_CIFAR(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_CIFAR(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_CIFAR(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_CIFAR(BasicBlock, [200, 200, 200])


def fixed_resnet20(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([3, 3, 3], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)


def fixed_resnet32(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([5, 5, 5], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)


def fixed_resnet44(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([7, 7, 7], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)


def fixed_resnet56(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([9, 9, 9], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)


def fixed_resnet110(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([18, 18, 18], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)


def fixed_resnet1202(k=1, fully_fixed=None, fixed_conv_params=None):
    return FixedResNet_CIFAR([200, 200, 200], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed)
