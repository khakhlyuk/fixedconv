from modules.resnet import *


def resnet20(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet110(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [18, 18, 18], num_classes=num_classes)


def resnet1202(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [200, 200, 200], num_classes=num_classes)


def fixed_resnet20(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([3, 3, 3], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)


def fixed_resnet32(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([5, 5, 5], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)


def fixed_resnet44(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([7, 7, 7], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)


def fixed_resnet56(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([9, 9, 9], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)


def fixed_resnet110(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([18, 18, 18], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)


def fixed_resnet1202(k, fully_fixed, fixed_conv_params, num_classes=10):
    return FixedResNet_CIFAR([200, 200, 200], k=k, fixed_conv_params=fixed_conv_params,
                             fully_fixed=fully_fixed, num_classes=num_classes)
