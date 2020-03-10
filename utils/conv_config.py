def get_fixed_conv_params(conv_type, n=None, sigma=None):
    """

    Args:
        conv_type (str): first letter specifies the type of filter to use.
        G - Gaussian
        M - Maxpooling
        A - Avgpooling

    Returns:

    """
    if conv_type == 'G':
        if n is None: n = 3
        if sigma is None: sigma = 0.8
        fixed_conv_params = {'conv_type': 'gaussian',
                             'kernel_size': n, 'sigma': sigma}

    elif conv_type == 'M':
        fixed_conv_params = {'conv_type': 'maxpool'}

    elif conv_type == 'A':
        fixed_conv_params = {'conv_type': 'avgpool'}

    else:
        raise RuntimeError("conv_type not defined")
    return fixed_conv_params
