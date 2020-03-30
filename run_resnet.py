from subprocess import run
# python -u run_resnet.py

cuda = 0  # which gpu to use
dataset = 'cifar10'

manualSeed = 99
workers = 2

for model in ['resnet20', 'resnet32']:
    # postfix = ''
    commands = [
        'python', '-u', 'train_resnet.py',
        '--dataset=' + dataset,
        '--model=' + model,
        '-c=' + str(cuda),
        '--workers=' + str(workers),
        '--manualSeed=' + str(manualSeed),
    ]
    run(commands)

for model in ['resnet20', 'resnet32']:
    f = True
    for k in [1, 2, 3]:
        for ff in [False, True]:
            # postfix = ''
            commands = [
                'python', '-u', 'train_resnet.py',
                '--dataset=' + dataset,
                '--model=' + model,
                '-k=' + str(k),
                '-c=' + str(cuda),
                '--workers=' + str(workers),
                '--manualSeed=' + str(manualSeed),
            ]
            if f: commands.append('-f')
            if ff: commands.append('--ff')
            run(commands)
