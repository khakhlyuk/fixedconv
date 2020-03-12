import torch.nn as nn
import math


# custom weights initialization called on netG and netD
def weights_init_dcgan(m, init_fixed=False):
    classname = m.__class__.__name__
    if classname in ['Conv2d', 'ConvTranspose2d']:  # only Convs, not FixedSeparable convs
        # init all convs of init_fixed, else only init trainable convs
        if init_fixed or m.weight.requires_grad is True:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def freeze(m):
    """Note: for a new dataset, it is advisable to leave BN unfrozen"""
    for layer in m.modules():
        for param in layer.parameters():
            param.requires_grad = False


def unfreeze(m):
    """ Note: Will make fixed convs learnable.
    """
    for layer in m.modules():
        for param in layer.parameters():
            param.requires_grad = True


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