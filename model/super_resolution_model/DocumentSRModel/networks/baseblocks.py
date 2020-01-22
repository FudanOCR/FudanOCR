# -*- coding:utf-8 -*-
import torch.nn as nn
from .snlayers.snconv2d import SNConv2d
from .snlayers.snlinear import SNLinear


DEBUG = False

# -------------------------
#  Common used CNN blocks
# -------------------------
'''
    Convolutional block
    * normalization options: batch instance
    * activation options: relu prelu lrelu tanh sigmoid
'''


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, activation='relu', norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_channels)
        elif self.norm == 'spectral':
            self.norm = None
            self.conv = SNConv2d(in_channels, out_channels,
                                 kernel_size, stride, padding, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)

        if self.norm is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.act(out)

        if DEBUG:
            print('Conv'+str(out.shape))
        return out


'''
    Dense block
    * normalization options: batch instance
    * activation options: relu prelu lrelu tanh sigmoid
'''


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(out_channels)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(out_channels)
        elif self.norm == 'spectral':
            self.norm = None
            self.conv = SNConv2d(in_channels, out_channels, 3, 1, 1, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)

        if self.norm is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.act(out)

        return out


'''
    Residual block
'''


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, kernel_size=3, stride=1, padding=1, bias=True,
                 dropout=False, padding_type=None, activation='relu', norm='batch'):
        super(ResidualBlock, self).__init__()

        conv_padding = 0
        self.pd = None

        if padding_type == 'reflect':
            self.pd = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pd = nn.ReplicationPad2d(1)
        else:
            conv_padding = 1

        self.conv1 = nn.Conv2d(num_filters, num_filters,
                               kernel_size, stride, conv_padding, bias=bias)
        self.conv2 = nn.Conv2d(num_filters, num_filters,
                               kernel_size, stride, conv_padding, bias=bias)

        self.dropout = nn.Dropout(0.5) if dropout else None

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(num_filters)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(num_filters)
        elif self.norm == 'spectral':
            self.norm = None
            self.conv = SNConv2d(num_filters, num_filters,
                                 kernel_size, stride, padding, bias=bias)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.pd is not None:
            out = self.conv1(self.pd(x))
        else:
            out = self.conv1(x)
        if self.norm is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.act(out)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.pd is not None:
            out = self.conv2(self.pd(out))
        else:
            out = self.conv2(out)
        if self.norm is not None:
            out = self.bn(out)
        out += residual
        if DEBUG:
            print('Res'+str(out.shape))
        return out


'''
    Deconvolution block
'''


class DeconvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=4, stride=2,
                 padding=1, bias=True, activation='relu', norm='batch'):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_size, out_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_size)
        else:
            self.bn = nn.BatchNorm2d(out_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.deconv(x)

        if self.norm is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.act(out)
        if DEBUG:
            print('Deconv'+str(out.shape))
        return out


'''
    PixelShuffler Block
'''


class PSBlock(nn.Module):
    def __init__(self, in_size, out_size, scale_factor, kernel_size=3, stride=1,
                 padding=1, bias=True, activation='relu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size * scale_factor**2,
                              kernel_size, stride, padding, bias=bias)
        self.ps = nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(out_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm2d(out_size)
        else:
            self.bn = nn.BatchNorm2d(out_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.ps(self.conv(x))

        if self.norm is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.act(out)
        return out


'''
    Upsample 2x Block
'''


class Upsample2xBlock(nn.Module):
    def __init__(self, in_size, out_size, bias=True, upsample='deconv',
                 activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2

        # Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(in_size, out_size, kernel_size=4,
                                        stride=2, padding=1, bias=bias,
                                        activation=activation, norm=norm)

        # Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(in_size, out_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(in_size, out_size, kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out
