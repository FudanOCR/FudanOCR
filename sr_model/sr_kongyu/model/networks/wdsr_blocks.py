import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    '''
    Convolutional block
    * normalization options: batch instance spectral weight
    * activation options: relu prelu lrelu tanh sigmoid
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 padding_type=None, activation='relu', normalization='batch'):
        super(ConvBlock, self).__init__()

        conv_padding = 0
        pd = None

        if padding_type == 'reflect':
            pd = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            pd = nn.ReplicationPad2d
        else:
            conv_padding = padding

        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride, conv_padding, bias=bias)

        norm = None
        if normalization == 'batch':
            norm = nn.BatchNorm2d(out_channels)
        elif normalization == 'instance':
            norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'weight':
            def wn(x): return nn.utils.weight_norm(x)
            conv = wn(conv)

        act = None
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        body = []

        if pd is not None:
            body.append(pd(padding))
        body.append(conv)
        if norm is not None:
            body.append(norm)
        if act is not None:
            body.append(act)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class ResidualBlock_A(nn.Module):
    def __init__(self, num_filters, kernel_size=3, res_scale=1, expand=3, bias=True,
                 padding_type=None, activation='relu', normalization='weight'):
        super(ResidualBlock_A, self).__init__()
        self.res_scale = res_scale

        conv_padding = 0
        pd = None

        if padding_type == 'reflect':
            pd = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            pd = nn.ReplicationPad2d
        else:
            conv_padding = kernel_size // 2

        conv1 = nn.Conv2d(num_filters, num_filters*expand,
                          kernel_size, 1, conv_padding, bias=bias)
        conv2 = nn.Conv2d(num_filters*expand, num_filters,
                          kernel_size, 1, conv_padding, bias=bias)

        norm = None
        if normalization == 'batch':
            norm = nn.BatchNorm2d
        elif normalization == 'instance':
            norm = nn.InstanceNorm2d
        elif norm == 'weight':
            def wn(x): return nn.utils.weight_norm(x)
            conv1 = wn(conv1)
            conv2 = wn(conv2)

            def wn(x): return nn.utils.weight_norm(x)
            conv1 = wn(conv1)
            conv2 = wn(conv2)

        act = None
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        body = []

        if pd is not None:
            body.append(pd(kernel_size//2))
        body.append(conv1)
        if norm is not None:
            body.append(norm(num_filters*expand))

        if act is not None:
            body.append(act)

        if pd is not None:
            body.append(pd(kernel_size//2))
        body.append(conv2)
        if norm is not None:
            body.append(norm(num_filters))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class ResidualBlock_B(nn.Module):
    def __init__(self, num_filters, kernel_size=3, res_scale=1, expand=3, linear=0.8, bias=True,
                 padding_type=None, activation='relu', normalization='weight'):
        super(ResidualBlock_B, self).__init__()
        self.res_scale = res_scale

        conv_padding = 0
        pd = None

        if padding_type == 'reflect':
            pd = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            pd = nn.ReplicationPad2d
        else:
            conv_padding = kernel_size // 2

        conv1 = nn.Conv2d(num_filters, num_filters*expand, 1, 1, 0, bias=bias)
        conv2_1 = nn.Conv2d(num_filters*expand,
                            int(num_filters*linear), 1, 1, 0, bias=bias)
        conv2_2 = nn.Conv2d(int(num_filters*linear), num_filters,
                            kernel_size, 1, conv_padding, bias=bias)

        norm = None
        if normalization == 'batch':
            norm = nn.BatchNorm2d
        elif normalization == 'instance':
            norm = nn.InstanceNorm2d
        elif norm == 'weight':
            def wn(x): return nn.utils.weight_norm(x)
            conv1 = wn(conv1)
            conv2_1 = wn(conv2_1)
            conv2_2 = wn(conv2_2)

        act = None
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        body = []

        body.append(conv1)
        if norm is not None:
            body.append(norm(num_filters*expand))

        if act is not None:
            body.append(act)

        body.append(conv2_1)
        if pd is not None:
            body.append(pd(kernel_size//2))
        body.append(conv2_2)
        if norm is not None:
            body.append(norm(num_filters))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class DeconvBlock(nn.Module):
    '''
    Deconvolution block
    '''

    def __init__(self, in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True,
                 activation='relu', normalization='batch'):
        super(DeconvBlock, self).__init__()
        deconv = nn.ConvTranspose2d(
            in_size, out_size, kernel_size, stride, padding, output_padding, bias=bias)

        norm = None
        if normalization == 'batch':
            norm = nn.BatchNorm2d(out_size)
        elif normalization == 'instance':
            norm = nn.InstanceNorm2d(out_size)
        elif normalization == 'weight':
            def wn(x): return nn.utils.weight_norm(x)
            deconv = wn(deconv)

        act = None
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        body = []

        body.append(deconv)
        if norm is not None:
            body.append(norm)
        if act is not None:
            body.append(act)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class PSBlock(nn.Module):
    '''
    PixelShuffler Block
    '''

    def __init__(self, in_size, out_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True,
                 activation='relu', normalization='batch'):
        super(PSBlock, self).__init__()
        conv = nn.Conv2d(in_size, out_size * scale_factor**2,
                         kernel_size, stride, padding, bias=bias)
        ps = nn.PixelShuffle(scale_factor)

        norm = None
        if normalization == 'batch':
            norm = nn.BatchNorm2d(out_size)
        elif normalization == 'instance':
            norm = nn.InstanceNorm2d(out_size)
        elif normalization == 'weight':
            def wn(x): return nn.utils.weight_norm(x)
            deconv = wn(conv)

        act = None
        if activation == 'relu':
            act = nn.ReLU(True)
        elif activation == 'prelu':
            act = nn.PReLU()
        elif activation == 'lrelu':
            act = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()

        body = []

        body.append(conv)
        body.append(ps)
        if norm is not None:
            body.append(norm)
        if act is not None:
            body.append(act)

        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class UpsampleBlock(nn.Module):
    '''
    Upsample Block
    '''
    def __init__(self, in_size, out_size, kernel_size=3, scale_factor=2, bias=True,
                            upsample='deconv', activation='relu', normalization='batch'):
        super(UpsampleBlock, self).__init__()

        # Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(in_size, out_size, kernel_size=kernel_size, stride=scale_factor, padding=int(kernel_size//2), output_padding=1,
                                        bias=bias, activation=activation, normalization=normalization)

        # Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(in_size, out_size, scale_factor=scale_factor, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2),
                                    bias=bias, activation=activation, normalization=normalization)

        # Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(in_size, out_size, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2),
                          bias=bias, activation=activation, normalization=normalization)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out
