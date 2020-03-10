import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname in ['Conv2d', 'ConvTranspose2d']:  # only Convs, not FixedSeparable convs
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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