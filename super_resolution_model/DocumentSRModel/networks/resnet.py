import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.baseblocks import *


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='prelu', upsample='ps',
                 padding_type='reflect', use_dropout=False, learn_residual=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True if norm == 'instance' else False
        model = [nn.ReflectionPad2d(3),
                 ConvBlock(input_nc, ngf, 7, 1, 0, bias=use_bias,
                           norm=norm, activation=activation)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ConvBlock(ngf * mult, ngf * mult * 2, 3, 2, 1, bias=use_bias,
                                norm=norm, activation=activation)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult, padding_type=padding_type,
                                    norm=norm, activation=activation,
                                    bias=use_bias, dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [Upsample2xBlock(ngf * mult, int(ngf * mult // 2), upsample=upsample,
                                      bias=use_bias, norm=norm, activation=activation)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
                  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.learn_residual:
            out = x + out
            out = torch.clamp(out, min=-1, max=1)
        return torch.sigmoid(out)


class Upscale4xResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='prelu', upsample='ps',
                 padding_type='reflect', use_dropout=False, learn_residual=False):
        assert(n_blocks >= 0)
        super(Upscale4xResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True if norm == 'instance' else False
        self.conv1 = ConvBlock(input_nc, ngf, 9, 1, 4, bias=use_bias,
                               norm=norm, activation=activation)

        # 2x size network blocks
        resblocks1L = []
        for i in range(n_blocks):
            resblocks1L += [ResidualBlock(ngf, padding_type=padding_type,
                                          norm=norm, activation=activation,
                                          bias=use_bias, dropout=use_dropout)]
        resblocks1L += [ConvBlock(ngf, ngf, 3, 1, 1, bias=use_bias,
                                  norm=norm, activation=None)]
        self.resblocks1L = nn.Sequential(*resblocks1L)

        self.upscale1L = Upsample2xBlock(ngf, ngf, upsample=upsample,
                                         bias=use_bias, norm=norm, activation=activation)
        self.outconv1L = ConvBlock(
            ngf, output_nc, 3, 1, 1, activation=None, norm=None)

        # 4x size network blocks
        resblocks2L = []
        for i in range(n_blocks):
            resblocks2L += [ResidualBlock(ngf, padding_type=padding_type,
                                          norm=norm, activation=activation,
                                          bias=use_bias, dropout=use_dropout)]
        self.resblocks2L = nn.Sequential(*resblocks2L)

        self.upscale2L = Upsample2xBlock(ngf, ngf, upsample=upsample,
                                         bias=use_bias, norm=norm, activation=activation)
        self.outconv2L = ConvBlock(
            ngf, output_nc, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x = self.conv1(x)

        out1L = self.resblocks1L(x)
        if self.learn_residual:
            out1L = x + out1L
        out1L = self.upscale1L(out1L)
        up2x = self.outconv1L(out1L)

        out2L = self.resblocks2L(out1L)
        if self.learn_residual:
            out2L = out1L + out2L
        out2L = self.upscale2L(out2L)
        up4x = self.outconv2L(out2L)

        return F.sigmoid(up2x), F.sigmoid(up4x)


class Upscale2xResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='prelu', upsample='ps',
                 padding_type='reflect', use_dropout=False, learn_residual=False):
        assert(n_blocks >= 0)
        super(Upscale2xResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True if norm == 'instance' else False
        self.conv1 = ConvBlock(input_nc, ngf, 9, 1, 4, bias=use_bias,
                               norm=norm, activation=activation)

        # 2x size network blocks
        resblocks = []
        for i in range(n_blocks):
            resblocks += [ResidualBlock(ngf, padding_type=padding_type,
                                        norm=norm, activation=activation,
                                        bias=use_bias, dropout=use_dropout)]
        resblocks += [ConvBlock(ngf, ngf, 3, 1, 1, bias=use_bias,
                                norm=norm, activation=None)]
        self.resblocks = nn.Sequential(*resblocks)

        self.upscale = Upsample2xBlock(ngf, ngf, upsample=upsample,
                                       bias=use_bias, norm=norm, activation=activation)
        self.outconv = ConvBlock(
            ngf, output_nc, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x = self.conv1(x)

        out = self.resblocks(x)
        if self.learn_residual:
            out = x + out
        out = self.upscale(out)
        out = self.outconv(out)

        return F.sigmoid(out)
