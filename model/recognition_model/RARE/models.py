import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from component.stn import SpatialTransformer


class RARE(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))
        self.opt = opt

        # self.stn = SpatialTransformer(self.opt)
        self.cnn = self.getCNN()
        self.rnn = self.getEncoder()
        # n_class,hidden_size,num_embedding,input_size
        # self.attention = Attention(self.n_class,256, 128,256)
        self.attention = Attention(256, 256, self.n_class, 128)


        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 21, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def stn(self, x):
        xs = self.localization(x)
        # print("size:", xs.size())
        xs = xs.view(-1, 10 * 4 * 21)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

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

        def convRelu(i, batchNormalization=False, leakyRelu=False):
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
        convRelu(2, True)
        convRelu(3, False)
        # 8 * 25
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # # 4 * 27
        convRelu(4, True)
        convRelu(5, False)
        # 4 * 27
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        # 2 * 29
        convRelu(6, True)
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

    def getEncoder(self):

        rnn = nn.Sequential(
            BLSTM(512, 256, 256),
            BLSTM(256, 256, 256)
        )
        return rnn

    # image, length, text, text_rev, test
    def forward(self, input, text_length, text, text_rev, test=False):

        input = self.stn(input)
        result = self.cnn(input)
        # (bs,512,1,5)

        # print('hi', result.size())
        B, C, H, W = result.size()
        assert H == 1, 'The height of the input image must be 1.'
        result = result.squeeze(2)
        result = result.permute(2, 0, 1)

        result = self.rnn(result)
        '''feature, text_length, test sign'''
        # result = self.attention(result,text,text_length, test)
        result = self.attention(result, text_length, text, test)
        return result


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128, CUDA=True):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        # self.fracPickup = fracPickup(CUDA=CUDA)

    def forward(self, prev_hidden, feats, cur_embeddings, test=False):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size

        feats_proj = self.i2h(feats.view(-1, nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1, nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(
            -1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT, nB)

        alpha = F.softmax(emition, 0)  # nB * nT

        if not test:
            # alpha_fp = self.fracPickup(alpha.unsqueeze(1).unsqueeze(2)).squeeze()
            context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha
        else:
            context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128, CUDA=True):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings, CUDA=CUDA)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes + 1, num_embeddings))
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.cuda = CUDA

    # targets is nT * nB
    def forward(self, feats, text_length, text, test=False):

        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert (input_size == nC)
        assert (nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        if not test:

            targets = torch.zeros(nB, num_steps + 1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                targets[i][1:1 + text_length.data[i]] = text.data[start_id:start_id + text_length.data[i]] + 1
                start_id = start_id + text_length.data[i]
            targets = Variable(targets.transpose(0, 1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB, hidden_size).type_as(feats.data))

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_hiddens[i] = hidden

            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                new_hiddens[start:start + length] = output_hiddens[0:length, b, :]
                start = start + length
                b = b + 1

            probs = self.generator(new_hiddens)
            return {
                'result': probs
            }

        else:

            hidden = Variable(torch.zeros(nB, hidden_size).type_as(feats.data))
            targets_temp = Variable(torch.zeros(nB).long().contiguous())
            probs = Variable(torch.zeros(nB * num_steps, self.num_classes))
            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                hidden2class = self.generator(hidden)
                probs[i * nB:(i + 1) * nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.num_classes).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                probs_res[start:start + length] = probs[b * num_steps:b * num_steps + length]
                start = start + length
                b = b + 1

            return {
                'result': probs_res
            }






class BLSTM(nn.Module):
    '''双向循环神经网络'''

    def __init__(self, nIn, nHidden, nOut):
        nn.Module.__init__(self)

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        '''The size of input must be [T,B,C]'''
        T, B, C = input.size()
        result, _ = self.rnn(input)
        result = result.view(T * B, -1)
        result = self.linear(result)
        result = result.view(T, B, -1)
        return result
