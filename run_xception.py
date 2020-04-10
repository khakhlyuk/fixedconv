from subprocess import run


cuda = 3  # which gpu to use
bs = 64
dataset = 'cifar100'
logs_path = 'xception_' + dataset

manualSeed = 99
workers = 0

for f in [False, True]:
    commands = [
        'python', '-u', 'train_xception.py',
        '--dataset=' + dataset,
        '-c=' + str(cuda),
        '--workers=' + str(workers),
        '--manualSeed=' + str(manualSeed),
        '--logs_path=' + logs_path,
        '--bs=' + str(bs),
    ]
    if f: commands.append('-f')
    run(commands)
