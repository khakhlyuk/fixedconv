from subprocess import run
# python -u run_dcgan.py

gpus = '1'  # comma separated numbers of which gpus to use
k_list = [(4, 4)]  # widening factors for G and D
fixedG = True
fixedD = True
net_type = 'A'

num_epochs = 20
workers = 2
manualSeed = 99
dataset = 'celeba'
dataroot = '/root/data/celeba/'

for kG, kD in k_list:

    postfix = ''

    outf = 'logs_dcgan/{}{}_{}{}_net{}_{}'.format(
        'fG' if fixedG else 'G', kG,
        'fD' if fixedD else 'D', kD,
        net_type,
        postfix)

    commands = [
        'python', '-u', 'train_dcgan.py',
        '--dataset=' + dataset,
        '--dataroot=' + dataroot,
        '--outf=' + outf,
        '--kG=' + str(kG),
        '--kD=' + str(kD),
        '--net_type=' + net_type,
        '--workers=' + str(workers),
        '--num_epochs=' + str(num_epochs),
        '--manualSeed=' + str(manualSeed),
        '--gpus=' + gpus,
        # "--netG_path=logs_dcgan/...pth",
        # # "--netD_path=logs_dcgan/...pth",
        # '--lr=0.00002',

    ]
    if fixedG: commands.append('--fixedG')
    if fixedD: commands.append('--fixedD')

    run(commands)
