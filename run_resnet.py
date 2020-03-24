#!/bin/bash

from subprocess import run
# python -u run_resnet.py

gpu = 3  # comma separated numbers of which gpus to use
dataset = 'cifar100'
conv_type = 'R'
ff = 'n'

manualSeed = 99
workers = 2

for model in ['resnet20']:
    # postfix = ''
    commands = [
        'python', '-u', 'train_resnet.py',
        '--dataset=' + dataset,
        '--model=' + model,
        '-c=' + str(gpu),
        '--workers=' + str(workers),
        '--manualSeed=' + str(manualSeed),
    ]

    run(commands)

for model in ['fixed_resnet20']:
    for k in [1, 2, 3]:
        for conv_type in ['R']:
            # postfix = ''
            commands = [
                'python', '-u', 'train_resnet.py',
                '--dataset=' + dataset,
                '--model=' + model,
                '--ff=' + ff,
                '-k=' + str(k),
                '--conv_type=' + conv_type,
                '-c=' + str(gpu),
                '--workers=' + str(workers),
                '--manualSeed=' + str(manualSeed),
            ]

            run(commands)


