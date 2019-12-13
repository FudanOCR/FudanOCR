import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from networks.baseblocks import *

class HED(nn.Module):
    def __init__(self, in_dims=3, out_dims=1):
        super(HED, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_dims, 64, 3, norm=None, activation='relu'),
            ConvBlock(64, 64, 3, norm=None, activation='relu')
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(64, 128, 3, norm=None, activation='relu'),
            ConvBlock(128, 128, 3, norm=None, activation='relu')
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(128, 256, 3, norm=None, activation='relu'),
            ConvBlock(256, 256, 3, norm=None, activation='relu'),
            ConvBlock(256, 256, 3, norm=None, activation='relu')
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(256, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu')
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu')
        )
        self.dsn1 = nn.Conv2d(64, out_dims, 1)
        self.dsn2 = nn.Conv2d(128, out_dims, 1)
        self.dsn3 = nn.Conv2d(256, out_dims, 1)
        self.dsn4 = nn.Conv2d(512, out_dims, 1)
        self.dsn5 = nn.Conv2d(512, out_dims, 1)
        self.fuse = nn.Conv2d(5, out_dims, 1)
        self.threshold = nn.Threshold(0.24, 0)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        d1 = self.dsn1(x1)
        d2 = F.interpolate(self.dsn2(x2), size=(h, w))
        d3 = F.interpolate(self.dsn3(x3), size=(h, w))
        d4 = F.interpolate(self.dsn4(x4), size=(h, w))
        d5 = F.interpolate(self.dsn5(x5), size=(h, w))
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = torch.sigmoid(d1)
        d2 = torch.sigmoid(d2)
        d3 = torch.sigmoid(d3)
        d4 = torch.sigmoid(d4)
        d5 = torch.sigmoid(d5)
        fuse = torch.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse

class HED_NUP(nn.Module):
    def __init__(self, in_dims=3, out_dims=1):
        super(HED, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_dims, 64, 3, norm=None, activation='relu'),
            ConvBlock(64, 64, 3, norm=None, activation='relu')
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(64, 128, 3, norm=None, activation='relu'),
            ConvBlock(128, 128, 3, norm=None, activation='relu')
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(128, 256, 3, norm=None, activation='relu'),
            ConvBlock(256, 256, 3, norm=None, activation='relu'),
            ConvBlock(256, 256, 3, norm=None, activation='relu')
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(256, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu')
        )
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu'),
            ConvBlock(512, 512, 3, norm=None, activation='relu')
        )
        self.dsn1 = nn.Conv2d(64, out_dims, 1)
        self.dsn2 = nn.Conv2d(128, out_dims, 1)
        self.dsn3 = nn.Conv2d(256, out_dims, 1)
        self.dsn4 = nn.Conv2d(512, out_dims, 1)
        self.dsn5 = nn.Conv2d(512, out_dims, 1)
        self.fuse = nn.Conv2d(5, out_dims, 1)

    def forward(self, x):
        h = x.size(2)
        w = x.size(3)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        d1 = self.dsn1(x1)
        d2 = self.dsn2(x2)
        d3 = self.dsn3(x3)
        d4 = self.dsn4(x4)
        d5 = self.dsn5(x5)
        fuse = self.fuse(torch.cat((d1, d2, d3, d4, d5), 1))

        d1 = torch.sigmoid(d1)
        d2 = torch.sigmoid(d2)
        d3 = torch.sigmoid(d3)
        d4 = torch.sigmoid(d4)
        d5 = torch.sigmoid(d5)
        fuse = torch.sigmoid(fuse)

        return d1, d2, d3, d4, d5, fuse
    
class HED_1L(nn.Module):
    def __init__(self, in_dims=3, out_dims=1):
        super(HED_1L, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_dims, 64, 3, norm=None, activation='relu'),
            ConvBlock(64, 64, 3, norm=None, activation='relu')
        )
        self.dsn1 = nn.Conv2d(64, out_dims, 1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        d1 = self.dsn1(x1)
        d1 = torch.sigmoid(d1)
        return d1

def hed_weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)