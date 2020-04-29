import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as init
from torch.nn.parameter import Parameter

capsule_dim = 8

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        # self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class Relu_Caps(nn.Module):
    def __init__(self, num_C, num_D, theta=0.2, eps=0.0001):
        super(Relu_Caps, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.theta = theta
        self.eps = eps

    def forward(self, x):
        x_caps = x.view(x.shape[0], self.num_C, self.num_D, x.shape[2], x.shape[3])
        x_length = torch.sqrt(torch.sum(x_caps * x_caps, dim=2))
        x_length = torch.unsqueeze(x_length, 2)
        x_caps = F.relu(x_length - self.theta) * x_caps / (x_length + self.eps)
        x = x_caps.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return x

class Caps_BN(nn.Module):
    '''
    Input variable N*CD*H*W
    First perform normal BN without learnable affine parameters, then apply a C group convolution to perform per-capsule
    linear transformation
    '''

    def __init__(self, num_C, num_D):
        super(Caps_BN, self).__init__()
        self.BN = nn.BatchNorm2d(num_C * num_D, affine=False)
        self.conv = nn.Conv2d(num_C * num_D, num_C * num_D, 1, groups=num_C)

        eye = torch.FloatTensor(num_C, num_D, num_D).copy_(torch.eye(num_D)).view(num_C * num_D, num_D,
                                                                                                  1, 1)
        self.conv.weight.data.copy_(eye)
        self.conv.bias.data.zero_()

    def forward(self, x):
        output = self.BN(x)
        output = self.conv(output)

        return output


class Caps_MaxPool(nn.Module):
    '''
    Input variable N*CD*H*W
    First get the argmax indices of capsule lengths, then tile the indices D time and apply the tiled indices to capsules
    '''

    def __init__(self, num_C, num_D, kernel_size, stride=None, padding=0, dilation=1):
        super(Caps_MaxPool, self).__init__()
        self.num_C = num_C
        self.num_D = num_D
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=True)

    def forward(self, x):
        B = x.shape[0]
        H, W = x.shape[2:]
        x_caps = x.view(B, self.num_C, self.num_D, H, W)
        x_length = torch.sum(x_caps * x_caps, dim=2)
        x_length_pool, indices = self.maxpool(x_length)
        H_pool, W_pool = x_length_pool.shape[2:]
        indices_tile = torch.unsqueeze(indices, 2).expand(-1, -1, self.num_D, -1, -1).contiguous()
        indices_tile = indices_tile.view(B, self.num_C * self.num_D, -1)
        x_flatten = x.view(B, self.num_C * self.num_D, -1)
        output = torch.gather(x_flatten, 2, indices_tile).view(B, self.num_C * self.num_D, H_pool, W_pool)

        return output


class Caps_Conv(nn.Module):
    def __init__(self, in_C, in_D, out_C, out_D, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(Caps_Conv, self).__init__()
        self.in_C = in_C
        self.in_D = in_D
        self.out_C = out_C
        self.out_D = out_D
        self.conv_D = nn.Conv2d(in_C * in_D, in_C * out_D, 1, groups=in_C, bias=False)
        self.conv_C = nn.Conv2d(in_C, out_C, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

        m = self.conv_D.kernel_size[0] * self.conv_D.kernel_size[1] * self.conv_D.out_channels
        self.conv_D.weight.data.normal_(0, math.sqrt(2. / m))
        n = self.conv_C.kernel_size[0] * self.conv_C.kernel_size[1] * self.conv_C.out_channels
        self.conv_C.weight.data.normal_(0, math.sqrt(2. / n))
        if bias:
            self.conv_C.bias.data.zero_()

    def forward(self, x):
        x = self.conv_D(x)
        x = x.view(x.shape[0], self.in_C, self.out_D, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.in_C, x.shape[3], x.shape[4])
        x = self.conv_C(x)
        x = x.view(-1, self.out_D, self.out_C, x.shape[2], x.shape[3])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.out_C * self.out_D, x.shape[3], x.shape[4])

        return x


class MyNet(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        '''first layer conv'''

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            Caps_MaxPool(64, 1, 2),

            Caps_Conv(64, 1, 128, capsule_dim, 3, 1, 1),
            Caps_BN(128,capsule_dim),
            Relu_Caps(128,capsule_dim),

            Caps_MaxPool(128, capsule_dim, 2),

            Caps_Conv(128, capsule_dim, 256, capsule_dim, 3, 1, 1),
            Caps_BN(256, capsule_dim),
            Relu_Caps(256, capsule_dim),

            Caps_Conv(256, capsule_dim, 256, capsule_dim, 3, 1, 1),
            Caps_BN(256, capsule_dim),
            Relu_Caps(256, capsule_dim),

            Caps_MaxPool(256, capsule_dim, (2, 2), (2, 1), (0, 1)),

            Caps_Conv(256, capsule_dim, 512, capsule_dim, 3, 1, 1),
            Caps_BN(512, capsule_dim),
            Relu_Caps(512, capsule_dim),

            Caps_Conv(512, capsule_dim, 512, capsule_dim, 3, 1, 1),
            Caps_BN(512, capsule_dim),
            Relu_Caps(512, capsule_dim),

            Caps_MaxPool(512, capsule_dim, (2, 2), (2, 1), (0, 1)),

            Caps_Conv(512, capsule_dim, 512, capsule_dim, 2,1,0),
            Caps_BN(512, capsule_dim),
            Relu_Caps(512, capsule_dim),

        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, 37),

        )


        '''capsule layer'''
        # self.conv2 =
        # self.bn2 =
        # self.relu = nn.ReLU(True)
        #
        # self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(True)
        #
        # self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(True)
        #
        # self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(True)

        # print("Initializing cnn net weights...")
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_normal_(m.weight.data)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()



    def forward(self,x):

        bs = x.size(0)

        result = self.cnn(x)
        result = result.view(bs,int(result.size(1)/capsule_dim),capsule_dim,result.size(2),result.size(3))

        result = torch.sqrt(torch.sum(result*result,dim=2))

        result = result.squeeze(2)  # b, c, w
        result = result.permute(2, 0, 1)  # w, b, c  -> (seq_len, batch_size, input_size)

        result = self.rnn(result)

        return {
            'result' : result
        }


if __name__ == '__main__':

    a = torch.Tensor(64,1,32,128)
    # net = Caps_Conv(1,1,16,32,1)
    # pool = Caps_MaxPool(16,32,(2,1),(2,1))
    # relu = Relu_Caps(16,32)
    # result = net(a)
    # result = pool(result)
    # result = relu(result)
    # print(result.size())
    net = MyNet()
    result = net(a)
    print(result.size())

    pass