import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class BasicBlock(nn.Module):
    '''
    构成ResNet的残差快
    '''

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_in, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 1), (2, 1))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(512)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

        # Official init from torch repo
        print("Initializing ResNet18 weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, returnhn=False):
        super(BidirectionalLSTM, self).__init__()

        # self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

        self.returnfn = returnhn

    def forward(self, input):
        recurrent, (hn, cn) = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        if self.returnfn == False:
            return output
        else:
            # hn 2,64,512
            hn = hn[0]  # 64, 512
            return output, hn


class Encoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 512, 512, False),
            BidirectionalLSTM(512, 512, 512, True),
        )

    def forward(self, input):
        x = input  # 64 512 6 40
        x = torch.max(x, 2)[0]
        x = x.permute(2, 0, 1).contiguous()  # 40*64*512
        x, hn = self.rnn(x)
        # print("Encoder里,hn的size为", hn.size())
        return x, hn


class AttentionCell(nn.Module):
    def __init__(self, opt):
        super(AttentionCell, self).__init__()
        self.score = nn.Linear(512, 1, bias=False)
        self.rnn = nn.GRUCell(512 + 128, 512)
        self.num_embeddings = 128
        self.hidden_size = 512
        self.H = 6
        self.W = 40

        '''给prev_hidden用的'''
        self.conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

    def forward(self, prev_hidden, conv_feats, conv_feats_origin, cur_embeddings, test=False):

        prev_hidden_temp = prev_hidden
        if len(prev_hidden.size()) == 2:
            prev_hidden = prev_hidden.unsqueeze(0).contiguous()  # 1 64 512

        nT = prev_hidden.size(0)
        nB = prev_hidden.size(1)
        nC = prev_hidden.size(2)

        # conv_feats 64 512 6 40
        # 64 512 240 -> 240 64 512 -> 240*64, 512
        conv_feats = conv_feats.view(nB, nC, -1).contiguous().permute(2, 0, 1).contiguous().view(-1, 512)
        conv_feats_origin = conv_feats_origin.view(nB, nC, -1).contiguous().permute(2, 0, 1).contiguous().view(-1, 512)

        # '''对feats进行加工'''
        # prev_hidden = prev_hidden.unsqueeze(0).contiguous().unsqueeze(3).contiguous()
        # # print("prev_hidden2的尺寸为", prev_hidden.size())
        #
        # prev_hidden = self.conv(prev_hidden)
        # prev_hidden = self.bn(prev_hidden)
        # prev_hidden = self.relu(prev_hidden)  # 64, 512 , 1 1
        # prev_hidden = prev_hidden.squeeze(3).contiguous().squeeze(2).contiguous() # 64,512

        # print(prev_hidden.size(), nB, nC)
        # 64, 512,
        prev_hidden = prev_hidden.expand(self.H * self.W, nB, nC).contiguous().view(-1, nC)
        # prev_hidden = prev_hidden.expand(nB, nC, self.H * self.W).contiguous()
        # print(prev_hidden.size())  # 15360,512

        # 对注意力机制进行改进
        emition = self.score(F.tanh(conv_feats + prev_hidden).view(-1, self.hidden_size)).view(self.H * self.W, nB)
        alpha = F.softmax(emition, 0)  # 240,64

        # print("!", alpha.size())  # 240,64
        # print("!", prev_hidden.size())

        context = (conv_feats_origin.view(self.H * self.W, nB,-1) * alpha.view(self.H * self.W, nB, 1).expand(self.H * self.W, nB, nC)).sum(
            0).squeeze(0)  # nB * nC
        # print("!", context.size())
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden_temp)
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(opt)

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))

        self.generator = nn.Linear(512, self.n_class)
        self.char_embeddings = Parameter(torch.randn(self.n_class + 1, 128))

        '''给conv_feats用的'''
        self.conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

    # targets is nT * nB
    def forward(self, conv_feats, init_state, text_length, text, test=False):
        '''
        feats 代表卷积神经网络最后一层的输入
        '''
        # conv_feats 64, 512, 6, 40
        nB = conv_feats.size(0)  # 64

        # conv_feats 先进行一波特征重组
        conv_feats_origin = conv_feats
        conv_feats = self.conv(conv_feats)
        conv_feats = self.bn(conv_feats)
        conv_feats = self.relu(conv_feats)  # conv_feats 64, 512, 6, 40

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()
        hidden_size = 512

        if not test:
            targets = torch.zeros(nB, num_steps + 1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                targets[i][1:1 + text_length.data[i]] = text.data[start_id:start_id + text_length.data[i]] + 1
                start_id = start_id + text_length.data[i]
            targets = Variable(targets.transpose(0, 1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(conv_feats.data))
            hidden = init_state

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, conv_feats, conv_feats_origin, cur_embeddings, test)
                output_hiddens[i] = hidden

            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(conv_feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                new_hiddens[start:start + length] = output_hiddens[0:length, b, :]
                start = start + length
                b = b + 1

            probs = self.generator(new_hiddens)
            return {
                'result': probs,
            }

        else:

            # hidden = Variable(torch.zeros(nB, hidden_size).type_as(feats.data))
            hidden = init_state
            # print("初始隐状态尺寸为", hidden.size())
            targets_temp = Variable(torch.zeros(nB).long().contiguous())
            probs = Variable(torch.zeros(nB * num_steps, self.n_class))
            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            '''返回注意力因子，做成一个字典的形式返回'''
            alphas = {}
            for i in range(nB):
                alphas[i] = []

            for i in range(num_steps):
                # print("num_steps",num_steps)
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, conv_feats, conv_feats_origin, cur_embeddings, test)
                # print("注意力向量维度:",alpha.size())  # (26,64)
                # print("隐状态尺寸为", hidden.size())

                for b in range(nB):
                    alphas[b].append(alpha.transpose(1, 0).contiguous().cpu().numpy()[b])

                # alphas = np.append(alphas,alpha.cpu().numpy())
                # alphas.append(alpha.cpu().numpy())

                hidden2class = self.generator(hidden)
                probs[i * nB:(i + 1) * nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.n_class).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.n_class).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.n_class).type_as(conv_feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                probs_res[start:start + length] = probs[b * num_steps:b * num_steps + length]
                start = start + length
                b = b + 1

            # print("注意力因子的尺寸为", alphas )
            return {
                'result': probs_res,
                # 'alphas': alphas
            }


class SAR(nn.Module):

    def __init__(self, opt=None):
        nn.Module.__init__(self)
        self.opt = opt

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))

        self.cnn = ResNet(num_in=opt.IMAGE.IMG_CHANNEL, block=BasicBlock, layers=[1, 2, 5, 3])  # (BS,6,40)
        self.encoder = Encoder()  # (40,BS,512)
        self.decoder = Attention(opt=opt)

    def forward(self, input, text_length, text, text_rev, test=False):
        conv_feats = self.cnn(input)
        encoder_feats, hn = self.encoder(conv_feats)
        # def forward(self, feats, text_length, text, init_state, test=False):
        x = self.decoder(conv_feats, hn, text_length, text, test)
        return x


if __name__ == '__main__':
    '''测试单元'''
    # import torch
    # AON = env.model
    sar = SAR()
    input = torch.Tensor(1, 1, 48, 160)
    output = sar(input)
    print("Size: ", output.size())  # 1, 512, 1, 5
