import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math

class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a)
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=1, P=4, stride=1, iters=3,
                 coor_add=False, w_shared=True):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int(((h - K )/stride)+ 1) # moein - changed from: oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        print(x.size())
        b, h, w, c = x.size()
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            # print(self.K)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


class CapsNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=64, B=8, C=16, D=16, E=10, K=1, P=4, iters=2):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=A,
                               kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=A, eps=0.001,
                                 momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=1, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        # self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters,
        #                                 coor_add=True, w_shared=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        # print("The output of conv_caps1 is: ", x.size())
        x = self.conv_caps2(x)
        # print("The output of conv_caps2 is: ",x.size() )
        return x
        # x = self.class_caps(x)
        # return x

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, returnhn=False):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
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
            # hn (2,64,512) -> (64,512)
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
        # print("382",x.size())
        x = torch.max(x, 2)[0]
        x = x.permute(2, 0, 1).contiguous()  # 40*64*512
        x, hn = self.rnn(x)
        # print('386',x.size())
        # print("Encoder里,hn的size为", hn.size())
        return x, hn


class AttentionCell(nn.Module):
    def __init__(self):
        super(AttentionCell, self).__init__()
        self.score = nn.Linear(512, 1, bias=False)
        self.rnn = nn.GRUCell(512 + 128, 512)
        self.num_embeddings = 128
        self.hidden_size = 512
        self.H = 22
        self.W = 22

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

        context = (conv_feats_origin.view(self.H * self.W, nB, -1) * alpha.view(self.H * self.W, nB, 1).expand(
            self.H * self.W, nB, nC)).sum(
            0).squeeze(0)  # nB * nC
        # print("!", context.size())
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden_temp)
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell()

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
            return probs

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
            return probs_res



class CAPSOCR(nn.Module):

    def __init__(self,opt):
        nn.Module.__init__(self)
        self.opt = opt

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))

        self.cnn = CapsNet(E=10)  # (BS,6,40)
        self.encoder = Encoder()  # (40,BS,512)
        self.decoder = Attention(opt)

        self.fc = nn.Linear(272,512,bias=True)
        self.relu = nn.ReLU()

    # def forward(self, input, text_length, text, text_rev, test=False):
    def forward(self, input,text_length, text, text_rev, test=False):

        bs = input.size(0)

        conv_feats = self.cnn(input)  #  8 22 22 272
        conv_feats = conv_feats.view(-1,conv_feats.size(3))
        conv_feats = self.relu(self.fc(conv_feats))
        conv_feats = conv_feats.view(bs,22,22,512)
        # print(conv_feats.size())
        # print('589',conv_feats.size())

        conv_feats = conv_feats.permute(0,3,1,2).contiguous()
        # print("584行",conv_feats.size())

        encoder_feats, hn = self.encoder(conv_feats)
        x = self.decoder(conv_feats, hn, text_length, text, test)
        return {
            'result' : x
        }

if __name__ == '__main__':
    device = torch.device("cuda")
    model = CAPSOCR()
    model = model.to(device)
    # model = model.cuda()

    image = torch.Tensor(1,1,100,100)
    image = image.cuda()
    output = model(image)

    print(model)
    print(output.size())
