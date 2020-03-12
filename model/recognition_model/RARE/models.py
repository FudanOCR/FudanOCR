import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class RARE(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET)) + 1
        self.opt = opt

        self.cnn = self.getCNN()
        self.rnn = self.getEncoder()
        # n_class,hidden_size,num_embedding,input_size
        self.attention = Attention(self.n_class,256, 128,256)

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
    def forward(self, input,text_length,text,text_rev,test=False):

        result = self.cnn(input)
        # (bs,512,1,5)

        # print('hi', result.size())
        B, C, H, W = result.size()
        assert H == 1, 'The height of the input image must be 1.'
        result = result.squeeze(2)
        result = result.permute(2, 0, 1)

        result = self.rnn(result)
        '''feature, text_length, test sign'''
        result = self.attention(result,text,text_length, test)
        return result


class AttentionCell(nn.Module):
    '''
    Define a special RNN.
    '''

    # self.attention_cell(hidden,feature,cur_embedding)
    # self.attention_cell = AttentionCell(input_size,num_embedding,hidden_size)
    def __init__(self,input_size,num_embeddings,hidden_size):
        nn.Module.__init__(self)

        self.h2h = nn.Linear(hidden_size,hidden_size)
        self.c2h = nn.Linear(input_size,hidden_size,bias=False)
        self.score = nn.Linear(hidden_size,1,bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)

        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size

    def forward(self,hidden,feature,embedding):
        '''
        hidden: B * H
        feature: T * B * C
        embedding: B * embedding_size
        '''

        T = feature.size(0)
        B = feature.size(1)
        C = feature.size(2)
        H = self.hidden_size

        feature_proj = self.c2h(feature.view(-1,H))  # T*B,H
        prev_hidden_proj = self.h2h(hidden).view(1,B,self.hidden_size).expand(T,B,self.hidden_size)\
            .contiguous().view(-1,H)   # T*B,H
        # emition = self.score(F.tanh(feature_proj + prev_hidden_proj).view(-1, H)).view(T,B) # T*B
        emition = self.score(F.tanh(feature_proj + prev_hidden_proj).view(-1, H)).view(T, B)
        alpha = F.softmax(emition,0)  # T*B
        # context = (feature * alpha.expand(T*B,C).contiguous().view(T,B,C)).sum(0).squeeze(0).view(B,C)
        context = (feature * alpha.view(T, B, 1).expand(T, B, C)).sum(0).squeeze(0)  # nB * nC
        context = torch.cat([context,embedding],1)
        cur_hidden = self.rnn(context,hidden)
        return cur_hidden, alpha

class Attention(nn.Module):

    def __init__(self,n_class,hidden_size,num_embedding,input_size):
        nn.Module.__init__(self)

        self.n_class = n_class
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embedding = num_embedding
        # input_size,num_embeddings,hidden_size
        self.attention_cell = AttentionCell(input_size,num_embedding,hidden_size)
        '''why +1?'''
        self.char_embeddings = Parameter(torch.randn(n_class+1,self.num_embedding))
        '''You need a generator to transform a embedded vector into character'''
        self.generator = nn.Linear(hidden_size,n_class)

    def forward(self,feature,text,text_length,test):

        T = feature.size(0)
        B = feature.size(1)
        C = feature.size(2)

        '''Define some assertions'''
        assert(self.input_size==C)
        assert(B==text_length.numel())

        '''最大迭代次数'''
        num_step = text_length.max()
        num_label = text_length.sum()
        hidden_size = self.hidden_size

        '''初试化隐藏状态'''
        hidden = Variable(torch.zeros(B,self.hidden_size))

        if not test:
            '''训练状态'''

            '''建立一个target区域'''
            target = torch.zeros(B,num_step+1).long().cuda()
            stard_id = 0
            for i in range(B):
                target[i][1:1+text_length[i]] = text[stard_id:stard_id+text_length[i]]
                stard_id += text_length[i]
            target = Variable(target.transpose(0,1).contiguous())

            hidden = Variable(torch.zeros(B,hidden_size).type_as(feature))
            output_hiddens = Variable(torch.zeros(num_step,B,hidden_size).type_as(feature))

            '''第一个step是什么？'''
            for i in range(num_step):
                cur_embedding = self.char_embeddings.index_select(0,target[i])
                hidden, alpha = self.attention_cell(hidden,feature,cur_embedding)
                output_hiddens[i] = hidden

            new_hidden = Variable(torch.zeros(num_label,hidden_size).type_as(feature))
            b = 0
            start = 0

            for length in text_length:
                new_hidden[start: start+length] = output_hiddens[0:length,b,:]
                b += 1
                start = start + length

            probs = self.generator(new_hidden)
            return probs

        else:
            '''测试状态'''

            hidden = Variable(torch.zeros(B, hidden_size).type_as(feature.data))
            targets_temp = Variable(torch.zeros(B).long().contiguous())
            probs = Variable(torch.zeros(B * num_step, self.n_class))

            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            for i in range(num_step):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feature, cur_embeddings)
                hidden2class = self.generator(hidden)
                probs[i * B:(i + 1) * B] = hidden2class
                '''Is max differential?'''
                _, targets_temp = hidden2class.max(1)
                '''why +1?'''
                targets_temp += 1

            probs = probs.view(num_step, B, self.n_class).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.n_class).contiguous()
            probs_res = Variable(torch.zeros(num_label, self.n_class).type_as(feature.data))
            b = 0
            start = 0

            '''
            At test procedure, is it possible for us to use length?
            '''
            for length in text_length.data:
                probs_res[start:start + length] = probs[b * num_step:b * num_step + length]
                start = start + length
                b = b + 1

            return probs_res

class BLSTM(nn.Module):
    '''双向循环神经网络'''

    def __init__(self, nIn, nHidden, nOut):
        nn.Module.__init__(self)

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        '''The size of input must be [T,B,C]'''
        T, B, C = input.size()
        result, _ = self.rnn(input)
        result = result.view(T * B, -1)
        result = self.linear(result)
        result = result.view(T, B, -1)
        return result
