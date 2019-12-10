import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt

from model.networks.baseblocks import *

class doubleConvBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(doubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, 1, 1, activation='relu', norm='batch'),
            ConvBlock(out_ch, out_ch, 3, 1, 1, activation='relu', norm='batch')
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class downBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            doubleConvBlock(in_ch, out_ch)
        )
    
    def forward(self, x):
        out = self.mpconv(x)
        return out

class upBlock(nn.Module):
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super(upBlock, self).__init__()
        
        if mode == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = doubleConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, incolordim=3, outcolordim=1):
        super(UNet, self).__init__()
        self.inc = doubleConvBlock(incolordim, 64)
        self.down1 = downBlock(64, 128)
        self.down2 = downBlock(128, 256)
        self.down3 = downBlock(256, 512)
        self.down4 = downBlock(512, 512)
        self.up1 = upBlock(1024, 256)
        self.up2 = upBlock(512, 128)
        self.up3 = upBlock(256, 64)
        self.up4 = upBlock(128, 64)
        self.outc = nn.Conv2d(64, outcolordim, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xup = self.down4(x4)

        xup = self.up1(xup, x4)
        xup = self.up2(xup, x3)
        xup = self.up3(xup, x2)
        xup = self.up4(xup, x1)
        xup = self.outc(xup)
        return F.softsign(xup)

def unet_weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()