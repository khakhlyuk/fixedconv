import numpy as np
import pandas as pd
import torch
import copy
from pathlib import Path


def format_number_km(x):
    if x > 1e6:
        return str(int(x / 1e6)) + 'M'
    elif x > 1e3:
        return str(int(x / 1e3)) + 'K'
    else:
        return str(x)


def num_params(model, count_fixed=False, display_all_modules=False,
               print_stats=False):
    """Counts number of parameters and layers in the model

    _exclude_from_layer_count contains names that shouldn't be counted
    as layers, shortcut layer, for example.

    Args:
        model (nn.Module):
        count_fixed (bool): if True, modules with fixed params
            (.requires_grad=False) are counted toward total_num_params too
        display_all_modules (bool): if True, prints info on all of the modules

    Returns:
        total_num_params, total_num_layers
    """
    _exclude_from_layer_count = ['shortcut']
    total_num_params = 0
    total_num_layers = 0
    for n, p in model.named_parameters():
        if p.requires_grad or count_fixed:
            num_params = 1
            for s in p.shape:
                num_params *= s
            if len(p.data.size()) > 1:
                excl_in_name = len(list(filter(lambda excl: excl in n, _exclude_from_layer_count))) > 0
                if not excl_in_name:
                    total_num_layers += 1
        else:
            num_params = 0
        if print_stats and display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    if print_stats:
        print("-" * 50)
        print("Total number of parameters: {:.2e}".format(total_num_params))
        print("-" * 50)
        print("Total number of layers: {}".format(total_num_layers))
    return total_num_params, total_num_layers


def format_scientific(value):
    is_numpy = type(value).__module__ == np.__name__

    if value is None:
        return 'None'
    elif is_numpy:
        type_name = type(value).__name__
        if 'float' in type_name or 'int' in type_name:
            return format_scientific(value.item())
    elif torch.is_tensor(value):
        return format_scientific(value.item())
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        return np.format_float_positional(value, precision=4, trim='-')
    return str(value)


def save_summary(path, values_dict, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    values_dict = copy.deepcopy(values_dict)
    for name, value in values_dict.items():
        values_dict[name] = format_scientific(value)

    if path.exists() and not overwrite:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=values_dict.keys())
    ser = pd.Series(values_dict)
    df = df.append(ser, ignore_index=True)
    df.to_csv(path_or_buf=path, mode='w', index=False)


def read_summary(path):
    return pd.read_csv(path)


def format_scientific_with_exp(value):
    is_numpy = type(value).__module__ == np.__name__

    if value is None:
        return 'None'
    elif is_numpy:
        type_name = type(value).__name__
        if 'float' in type_name or 'int' in type_name:
            return format_scientific(value.item())
    elif torch.is_tensor(value):
        return format_scientific(value.item())
    elif isinstance(value, int):
        if abs(value) > 1e3 or abs(value) < 1e-3:
            return np.format_float_scientific(value, precision=3, trim='-',
                                              exp_digits=1)
        else:
            return str(value)
    elif isinstance(value, float):
        if abs(value) > 1e3 or abs(value) < 1e-3:
            return np.format_float_scientific(value, precision=3, trim='-',
                                              exp_digits=1)
        else:
            return np.format_float_positional(value, precision=4, trim='-')
    return str(value)
