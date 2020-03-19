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


class BLSTM2(nn.Module):
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
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cluenet(x)
        x = x.view(-1, 64)
        x = self.linear1(x)
        x = self.relu(x)
        x = x.view(-1, 512)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1,4, 23)
        x = F.softmax(x, 1)
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

        batch_size = clue.size(0)

        clue = clue.view(4,batch_size,23)
        c1, c2, c3, c4 = clue
        # print("c1:", c1.size())
        c1 = c1.view(batch_size,23).expand(512,batch_size,23).contiguous().view(23,batch_size,512)
        c2 = c2.view(batch_size,23).expand(512,batch_size,23).contiguous().view(23,batch_size,512)
        c3 = c3.view(batch_size,23).expand(512,batch_size,23).contiguous().view(23,batch_size,512)
        c4 = c4.view(batch_size,23).expand(512,batch_size,23).contiguous().view(23,batch_size,512)

        # print("c1:", c1.size())
        # print("feats1:", feat1.size())

        combine = F.tanh(c1 * feat1 + c2 * feat2 + c3 * feat3 + c4 * feat4)
        combine = combine.view(23,batch_size,512)
        return combine


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
            return probs

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

            return probs_res



class Decoder(nn.Module):
    def __init__(self,opt):
        nn.Module.__init__(self)

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))

        self.blstm = BLSTM(512,256)
        self.attention = Attention(input_size=512, hidden_size=256, num_classes=self.n_class, num_embeddings=128)

    def forward(self, x,text_length, text, test):

        x = self.blstm(x)
        x = self.attention(x, text_length, text, test)
        return x


class AON(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.opt = opt
        self.bcnn = BCNN()
        self.cluenet = ClueNet()
        self.multidirectionnet = MultiDirectionNet()
        self.fg = FG()
        self.decoder = Decoder(self.opt)

        print("Initializing cnn net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image, text_length, text,text_rev,test=False):
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
        # print(combine_feats.size(),'尺寸')

        '''Decoder part'''
        output = self.decoder(combine_feats,text_length, text, test)



        return output
