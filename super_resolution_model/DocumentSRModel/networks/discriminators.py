import torch
import torch.nn as nn

from networks.baseblocks import *

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        
        seq = [ConvBlock(input_nc, ndf, 4, 2, 1,\
                        activation='lrelu', norm=None)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers+1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stride = 1 if n == n_layers else 2
            seq.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, 4, stride, 1,\
                                activation='lrelu', norm='batch'))

        seq.append(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 2))
        if use_sigmoid:
            seq.append(nn.Sigmoid())

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.size(0), -1)
        out = torch.mean(out, 1)
        return out