from subprocess import run
# python -u run_dcgan.py

# --netG_path
# --netD_path

gpus = '3'  # comma separated numbers of which gpus to use
k_list = [(3, 3)]  # widening factors for G and D
fixedG = True
fixedD = True
option_fD = 'A'  # 'A' for FC layer, 'B' for avgpool + 1x1
sigma_list = [0.6, 1.0, 0.4, 1.5, 0.2, 2.0]  # sigmas for gaussian kernel

num_epochs = 20
workers = 4
manualSeed = 99
dataset = 'celeba'
dataroot = '/root/data/celeba/'

for kG, kD in k_list:

    for sigma in sigma_list:
        postfix = 'opt={}_sigma={}_'.format(option_fD, sigma)

        outf = 'logs_dcgan/{}{}_{}{}_{}'.format(
            'fG' if fixedG else 'G', kG,
            'fD' if fixedD else 'D', kD,
            postfix)

        commands = [
            'python', '-u', 'train_dcgan.py',
            '--dataset=' + dataset,
            '--dataroot=' + dataroot,
            '--outf=' + outf,
            '--kG=' + str(kG),
            '--kD=' + str(kD),
            '--workers=' + str(workers),
            '--num_epochs=' + str(num_epochs),
            '--manualSeed=' + str(manualSeed),
            '--gpus=' + gpus,
            '--option_fD=' + option_fD,
            '--sigma=' + str(sigma),
        ]
        if fixedG: commands.append('--fixedG')
        if fixedD: commands.append('--fixedD')

        run(commands)
