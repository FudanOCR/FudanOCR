import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def flip(x, dim):
    '''Flip a tensor according to the dim'''
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class BLSTM(nn.Module):
    '''双向循环神经网络'''

    def __init__(self, nIn, nHidden):
        nn.Module.__init__(self)

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        # self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        '''The size of input must be [T,B,C]'''
        T, B, C = input.size()
        result, _ = self.rnn(input)
        result = result.view(T, B, -1)
        # result = self.linear(result)
        # result = result.view(T, B, -1)
        return result


class BCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.bcnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1)),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )

    def forward(self, x):
        x = self.bcnn(x)
        return x


class ClueNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.cluenet = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (1, 1)),
        )
        self.linear1 = nn.Linear(64, 23)
        self.linear2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.cluenet(x)
        x = x.view(512, 64)
        x = self.linear1(x)
        x = x.view(23, 512)
        x = self.linear2(x)
        x = x.view(4, 23)
        x = F.softmax(x, 0)
        return x


class MultiDirectionNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.blstm = BLSTM(512, 256)

        self.multi = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (1, 0)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)),
        )

    def forward(self, x):
        x = self.multi(x)

        B, C, H, W = x.size()
        assert H == 1, "The height must be 1"
        x = x.squeeze(2)
        x = x.view(W, B, C)
        x = self.blstm(x)
        return x


class FG(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, feat1, feat2, feat3, feat4, clue):
        c1, c2, c3, c4 = clue
        print("c1:", c1.size())
        c1 = c1.view(1,23).expand(512,1,23).contiguous()
        c2 = c2.view(1,23).expand(512,1,23).contiguous()
        c3 = c3.view(1,23).expand(512,1,23).contiguous()
        c4 = c4.view(1,23).expand(512,1,23).contiguous()

        print("c1:", c1.size())
        print("feats1:", feat1.size())

        combine = F.tanh(c1 * feat1 + c2 * feat2 + c3 * feat3 + c4 * feat4)
        return combine


class AON(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.opt = opt
        self.bcnn = BCNN()
        self.cluenet = ClueNet()
        self.multidirectionnet = MultiDirectionNet()
        self.fg = FG()

    def getFG(self):
        pass

    def getDecoder(self):
        pass

    def forward(self, image):
        '''BCNN part'''
        x_left_right = self.bcnn(image)
        x_top_down = x_left_right.permute(0, 1, 3, 2)

        '''AON part'''
        feats_left_right = self.multidirectionnet(x_left_right)
        feats_right_left = flip(feats_left_right, 0)
        feats_top_down = self.multidirectionnet(x_top_down)
        feats_down_top = flip(feats_top_down, 0)
        clues = self.cluenet(x_left_right)

        '''FG part'''
        combine_feats = self.fg(feats_left_right, feats_right_left, feats_top_down, feats_down_top, clues)

        return combine_feats
