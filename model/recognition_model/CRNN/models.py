import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class CRNN(nn.Module):
    def __init__(self,opt):
        super(CRNN, self).__init__()
        self.opt = opt

        '''alphabet'''
        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET)) + 1

        '''cnn'''
        self.cnn = self.getCNN()

        '''rnn'''
        self.rnn = self.getRNN()


    def getCNN(self):

        '''cnn'''
        nc = self.opt.IMAGE.IMG_CHANNEL
        '''
        nm: chanel number
        ks: kernel size
        ps: padding size
        ss: stride size
        '''
        nm = [64, 128, 256, 256, 512, 512, 512]
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]




        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False,leakyRelu=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # 32 * 100
        convRelu(0, False)
        # 32 * 100
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        # 16 * 50
        convRelu(1, False)
        # 16 * 50
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        # 8 * 25
        convRelu(2, False)
        convRelu(3, False)
        # 8 * 25
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((1,2), (2,1)))
        # # 4 * 27
        convRelu(4, True)
        convRelu(5, True)
        # 4 * 27
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((1,2), (2,1)))
        # 2 * 29
        convRelu(6, False)
        # 1 * ?
        # 也就是说，当图片的高为32时，经过卷积层之后，输出的特征图维度的高将变为1

        print("Initializing cnn net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return cnn

    def getRNN(self):
        rnn = nn.Sequential(
            BidirectionalLSTM(512,256,256),
            BidirectionalLSTM(256, 256, self.n_class),
        )

        return rnn


    def forward(self, input):
        conv = self.cnn(input)

        b, c, h, w = conv.size()
        assert h==1

        conv = conv.squeeze(2) # b, c, w
        conv = conv.permute(2, 0, 1) # w, b, c  -> (seq_len, batch_size, input_size)

        rnn_result = self.rnn(conv)
        return rnn_result



class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        # self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output