import torch.nn as nn


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