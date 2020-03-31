from subprocess import run
# python -u val_resnet.py

cuda = 0  # which gpu to use
dataset = 'cifar10'
logs_path = 'logs_resnet' + '_' + dataset

manualSeed = 99
workers = 0

for model in ['resnet20', 'preact_resnet20']:
    commands = [
        'python', '-u', 'validate_resnet.py',
        '--dataset=' + dataset,
        '--model=' + model,
        '-c=' + str(cuda),
        '--workers=' + str(workers),
        '--manualSeed=' + str(manualSeed),
        '--logs_path=' + logs_path,
    ]
    run(commands)

for model in ['resnet20', 'preact_resnet20']:
    f = True
    for k in [1, 3]:
        for ff in [False, True]:
            commands = [
                'python', '-u', 'validate_resnet.py',
                '--dataset=' + dataset,
                '--model=' + model,
                '-k=' + str(k),
                '-c=' + str(cuda),
                '--workers=' + str(workers),
                '--manualSeed=' + str(manualSeed),
                '--logs_path=' + logs_path,
            ]
            if f: commands.append('-f')
            if ff: commands.append('--ff')
            run(commands)
