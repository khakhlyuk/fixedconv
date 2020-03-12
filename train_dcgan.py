import argparse
import os
from functools import partial
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt


import modules.dcgan as dcgan
from modules.functions import weights_init_dcgan
from modules.fixedconv import get_fixed_conv_params


conv_type_names = ['G', 'A']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    help='celeba | cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--outf', default='.',
                    help='folder to output images and model checkpoints')
parser.add_argument('--kG', type=int,
                    help='Widening factor for the Generator.', default=1)
parser.add_argument('--kD', type=int,
                    help='Widening factor for the Discriminator.', default=1)
parser.add_argument('--fixedG', action='store_true', help='Use Fixed G')
parser.add_argument('--fixedD', action='store_true', help='Use Fixed D')
parser.add_argument('--freezeG', action='store_true', help="Freeze G and don't train it")
parser.add_argument('--freezeD', action='store_true', help="Freeze D and don't train it")
parser.add_argument('--conv_type', default='G',
                    choices=conv_type_names,
                    help='Conv types. Gaussian or AvgPool'
                         'Used for fixed nets only.'
                         'Choices: ' + ' | '.join(conv_type_names))
parser.add_argument('--sigma', type=float, default=0.75,
                    help='Parameter sigma for the gaussian kernel.')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=128,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--gpus', default='0',
                    help='Comma separated list of cuda GPUs to train on')
parser.add_argument('--netG_path', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD_path', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom',
                    help='comma separated list of classes for the lsun data set')
parser.add_argument('--option_fD', default='A',
                    help='Option for Fixed Discriminator. Check docs. ')

args = parser.parse_args()
print(args)

outf = args.outf
try:
    os.makedirs(outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if args.dataroot is None and str(args.dataset).lower() != 'fake':
    raise ValueError(
        "`dataroot` parameter is required for dataset \"%s\"" % args.dataset)

if args.dataset in ['celeba', 'imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    nc = 3
elif args.dataset == 'lsun':
    classes = [c + '_train' for c in args.classes.split(',')]
    dataset = dset.LSUN(root=args.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5)),
                        ]))
    nc = 3
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)),
                           ]))
    nc = 3

elif args.dataset == 'mnist':
    dataset = dset.MNIST(root=args.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(args.imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1

elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
    nc = 3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True,
                                         num_workers=args.workers)

gpus = [('cuda:' + x) for x in args.gpus.split(',')]
device = torch.device(gpus[0] if torch.cuda.is_available() else "cpu")

nz = args.nz * args.kG
ngf = args.ngf * args.kG
ndf = args.ndf * args.kD

nz_hw = 4 if args.fixedG else 1  # height/width of noise vector

# Gaussian Layer Parameters
fixed_conv_params = get_fixed_conv_params(
    args.conv_type, bilin_interpol=True, n=4, sigma=args.sigma)

# Creating G and D
if args.fixedG:
    netG = dcgan.FixedGenerator(nz, ngf, nc, fixed_conv_params).to(device)
else:
    netG = dcgan.Generator(nz, ngf, nc).to(device)
if args.fixedD:
    netD = dcgan.FixedDiscriminator(nc, ndf, fixed_conv_params, args.option_fD).to(device)
else:
    netD = dcgan.Discriminator(nc, ndf).to(device)

# CHANGE if you want
init_fixed = False

netG.apply(partial(weights_init_dcgan, init_fixed=init_fixed))
netD.apply(partial(weights_init_dcgan, init_fixed=init_fixed))

if args.netG_path != '':
    netG.load_state_dict(torch.load(args.netG_path))

if args.netD_path != '':
    netD.load_state_dict(torch.load(args.netD_path))

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, nz_hw, nz_hw, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# Lists to keep track of progress
G_losses = []
D_losses = []
iters = 0

# Change if you want to checkpoint more often
checkpoint_every = args.num_epochs // 2

for epoch in range(args.num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, nz_hw, nz_hw, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        D_loss = errD.item() if not args.freezeD else 0
        G_loss = errG.item() if not args.freezeG else 0
        D_losses.append(D_loss)
        G_losses.append(G_loss)

        # Output training stats
        if i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.num_epochs, i, len(dataloader),
                     D_loss, G_loss, D_x, D_G_z1, D_G_z2))

        iters += 1

    # Check how the generator is doing by saving G's output on fixed_noise
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    vutils.save_image(fake[:64],
                      '%s/fake_samples_epoch_%03d.png' % (
                          outf, epoch),
                      padding=2,
                      normalize=True,)

    # do checkpointing (every n epochs and on the last epoch)
    if ((epoch + 1) % checkpoint_every == 0) or (epoch == args.num_epochs-1):
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))

# Grab a batch of real images from the dataloader and save it
real = next(iter(dataloader))[0]
vutils.save_image(real[:64],
                  '%s/real_samples.png' % outf,
                  normalize=True)

# Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('%s/Losses.png' % outf)
