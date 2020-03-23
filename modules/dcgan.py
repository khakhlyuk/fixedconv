import torch.nn as nn
from modules.fixedconv import FixedSeparableConv2d, FixedSeparableConvTrans2d
from modules.fixedconv import GaussianLayerTrans, RandomConvTrans2d, RandomConv2d

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # x is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class FixedGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, fixed_conv_params, net_type, k=1):
        super(FixedGenerator, self).__init__()

        if net_type == "A":
            ngf = ngf * k
            self.main = nn.Sequential(
                # input is Z. nz x 1 x 1
                nn.Conv2d(nz, ngf * 8, kernel_size=1, stride=1, padding=0,
                          bias=False),
                RandomConvTrans2d(ngf * 8, 4, 1, 0, 0, 0, 0.02),

                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4

                FixedSeparableConvTrans2d(ngf * 8, ngf * 4, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8

                FixedSeparableConvTrans2d(ngf * 4, ngf * 2, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16

                FixedSeparableConvTrans2d(ngf * 2, ngf * 1, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf * 1),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32

                FixedSeparableConvTrans2d(ngf * 1, nc, 2, fixed_conv_params),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        elif net_type == "B":
            ngf_nok = ngf
            ngf = ngf * k
            self.main = nn.Sequential(
                # input is Z. nz x 1 x 1

                nn.ConvTranspose2d( nz, ngf_nok * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf_nok * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4

                FixedSeparableConvTrans2d(ngf_nok * 8, ngf * 4, 2, fixed_conv_params),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8

                FixedSeparableConvTrans2d(ngf * 4, ngf * 2, 2, fixed_conv_params),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16

                FixedSeparableConvTrans2d(ngf * 2, ngf * 1, 2, fixed_conv_params),
                nn.BatchNorm2d(ngf * 1),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32

                FixedSeparableConvTrans2d(ngf * 1, nc, 2, fixed_conv_params),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        elif net_type == "C":
            ngf_nok = ngf
            ngf = ngf * k
            self.main = nn.Sequential(
                # input is Z. nz x 1 x 1

                nn.ConvTranspose2d(nz, ngf_nok * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf_nok * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4

                FixedSeparableConvTrans2d(ngf_nok * 8, ngf * 4, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8

                FixedSeparableConvTrans2d(ngf * 4, ngf * 2, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16

                FixedSeparableConvTrans2d(ngf * 2, ngf_nok * 1, 2,
                                          fixed_conv_params),
                nn.BatchNorm2d(ngf_nok * 1),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32

                nn.ConvTranspose2d(ngf_nok * 1, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, x):
        return self.main(x)


class FixedDiscriminator(nn.Module):
    def __init__(self, nc, ndf, fixed_conv_params, net_type, k=1):
        super(FixedDiscriminator, self).__init__()

        if net_type == "A":
            self.main = nn.Sequential(
                # x is (nc) x 64 x 64
                FixedSeparableConv2d(nc, ndf * 1, 2, fixed_conv_params),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32

                FixedSeparableConv2d(ndf * 1, ndf * 2, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16

                FixedSeparableConv2d(ndf * 2, ndf * 4, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8

                FixedSeparableConv2d(ndf * 4, ndf * 8, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4

                # just to use an AvgPooling instead of a fixed conv
                # nn.AdaptiveAvgPool2d(1),
                RandomConv2d(ndf * 8, 4, 1, 0, 0, 0.02),
                nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.Sigmoid()
            )
        elif net_type == "B":
            ndf_nok = ndf
            ndf = ndf * k
            self.main = nn.Sequential(
                # x is (nc) x 64 x 64
                nn.Conv2d(nc, ndf_nok * 1, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32

                FixedSeparableConv2d(ndf_nok * 1, ndf * 2, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16

                FixedSeparableConv2d(ndf * 2, ndf * 4, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8

                FixedSeparableConv2d(ndf * 4, ndf * 8, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4

                # just to use an AvgPooling instead of a fixed conv
                # nn.AdaptiveAvgPool2d(1),
                RandomConv2d(ndf * 8, 4, 1, 0, 0, 0.02),
                nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Sigmoid()
            )
        elif net_type == "C":
            ndf_nok = ndf
            ndf = ndf * k
            self.main = nn.Sequential(
                # x is (nc) x 64 x 64
                nn.Conv2d(nc, ndf_nok * 1, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32

                FixedSeparableConv2d(ndf_nok * 1, ndf * 2, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16

                FixedSeparableConv2d(ndf * 2, ndf * 4, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8

                FixedSeparableConv2d(ndf * 4, ndf_nok * 8, 2, fixed_conv_params),
                nn.BatchNorm2d(ndf_nok * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4

                nn.Conv2d(ndf_nok * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )


    def forward(self, x):
        return self.main(x)



# """ A2 Version. 4x4 noise, conv2d in the end of D"""
# class FixedGenerator(nn.Module):
#     def __init__(self, nz, ngf, nc, fixed_conv_params):
#         super(FixedGenerator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z. nz x 4 x 4
#             FixedSeparableConvTrans2d(nz, ngf * 8, 1, fixed_conv_params),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#
#             FixedSeparableConvTrans2d(ngf * 8, ngf * 4, 2, fixed_conv_params),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#
#             FixedSeparableConvTrans2d(ngf * 4, ngf * 2, 2, fixed_conv_params),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#
#             FixedSeparableConvTrans2d(ngf * 2, ngf * 1, 2, fixed_conv_params),
#             nn.BatchNorm2d(ngf * 1),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#
#             FixedSeparableConvTrans2d(ngf * 1, nc, 2, fixed_conv_params),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, x):
#         # ensure that input is bsxnzx4x4
#         assert x.shape[2] == 4 and x.shape[3] == 4
#         return self.main(x)
#
#
# """ A2 Version. 4x4 noise, conv2d in the end of D"""
# class FixedDiscriminator(nn.Module):
#     def __init__(self, nc, ndf, fixed_conv_params, option='A'):
#         super(FixedDiscriminator, self).__init__()
#         self.main = nn.Sequential(
#             # x is (nc) x 64 x 64
#             FixedSeparableConv2d(nc, ndf, 2, fixed_conv_params),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#
#             FixedSeparableConv2d(ndf * 1, ndf * 2, 2, fixed_conv_params),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#
#             FixedSeparableConv2d(ndf * 2, ndf * 4, 2, fixed_conv_params),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#
#             FixedSeparableConv2d(ndf * 4, ndf * 8, 2, fixed_conv_params),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#         )
#
#         # original version. Equivalent to FC layer
#         if option == 'A':
#             self.head = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
#         # Global AvgPool + 1x1 conv
#         elif option == 'B':
#             self.head = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(1),
#                 nn.Conv2d(ndf * 8, 1, 1, 1, 0, bias=False)
#             )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         return self.sigmoid(self.head(self.main(x)))

