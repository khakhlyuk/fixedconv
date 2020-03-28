#!/bin/bash

from subprocess import run
# python -u run_resnet.py

cuda = 3  # which gpu to use
dataset = 'cifar100'
conv_type = 'R'

manualSeed = 99
workers = 2

# for model in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
#     # postfix = ''
#     commands = [
#         'python', '-u', 'train_resnet.py',
#         '--dataset=' + dataset,
#         '--model=' + model,
#         '-c=' + str(cuda),
#         '--workers=' + str(workers),
#         '--manualSeed=' + str(manualSeed),
#     ]
#
#     run(commands)

for model in ['fixed_resnet44', 'fixed_resnet56', 'fixed_resnet110']:
    for k in [3]:
        for ff in ['n', 'y']:
            for conv_type in ['R']:
                # postfix = ''
                commands = [
                    'python', '-u', 'train_resnet.py',
                    '--dataset=' + dataset,
                    '--model=' + model,
                    '--ff=' + ff,
                    '-k=' + str(k),
                    '--conv_type=' + conv_type,
                    '-c=' + str(cuda),
                    '--workers=' + str(workers),
                    '--manualSeed=' + str(manualSeed),
                ]

                run(commands)

# for model in ['fixed_resnet44', 'fixed_resnet56', 'fixed_resnet110']:
#     for k in [3]:
#         for conv_type in ['R']:
#             # postfix = ''
#             commands = [
#                 'python', '-u', 'train_resnet.py',
#                 '--dataset=' + dataset,
#                 '--model=' + model,
#                 '--ff=' + ff,
#                 '-k=' + str(k),
#                 '--conv_type=' + conv_type,
#                 '-c=' + str(cuda),
#                 '--workers=' + str(workers),
#                 '--manualSeed=' + str(manualSeed),
#             ]
#
#             run(commands)

