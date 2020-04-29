import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from component.stn import SpatialTransformer
from torchvision.ops import roi_align

planes = 16
caps_size = 4
num_caps = 64


class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.conv0 = nn.Conv2d(1, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(planes)
        self.pool0 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv1 = nn.Conv2d(planes , planes * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)

        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        self.conv4 = nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * 8)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # self.conv5 = nn.Conv2d(planes * 8, planes * 8, kernel_size=3, stride=1, padding=1, bias=False),
        # self.bn5 = nn.BatchNorm2d(planes * 8),
        # self.pool5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1)),



        # self.pool6 = nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
        # nn.BatchNorm2d(planes * 8),
        # nn.ReLU(True),
        # nn.Conv2d(planes * 8, planes * 8, kernel_size=2, stride=1, padding=0, bias=False),
        # nn.BatchNorm2d(planes * 8),
        # nn.ReLU(True),

    def forward(self, input):

        # input = self.conv0(input)
        # input = self.bn0(input)
        # input = F.relu(input)
        # print(input)
        # print(input.size())

        feature_pyramid = []

        input = F.relu(self.bn0(self.conv0(input)))
        input = self.pool0(input)

        # print(input.size())
        feature_pyramid.append(input)

        input = F.relu(self.bn1(self.conv1(input)))
        input = self.pool1(input)

        input = F.relu(self.bn2(self.conv2(input)))

        # print(input.size())
        feature_pyramid.append(input)

        input = F.relu(self.bn3(self.conv3(input)))
        input = self.pool3(input)

        # print(input.size())
        feature_pyramid.append(input)

        input = F.relu(self.bn4(self.conv4(input)))
        input = self.pool4(input)


        # print("CNN层的输出结果为",input.size())
        # input = F.relu(self.bn5(self.conv5(input)))

        # for i,feature in enumerate(output):
        #     print("特征金字塔{0}的尺寸为".format(i),feature.size())


        return input, feature_pyramid


class RARE(nn.Module):

    def __init__(self, opt):
        nn.Module.__init__(self)

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))
        self.opt = opt

        # self.stn = SpatialTransformer(self.opt)
        self.cnn = CNN()
        self.rnn = self.getEncoder()
        # n_class,hidden_size,num_embedding,input_size
        # self.attention = Attention(self.n_class,256, 128,256)
        self.attention = Attention(256, 256, self.n_class, 128)

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # ========= ConvCaps Layers
        for d in range(1, 2):
            '''自回归模型'''

            self.conv_layers.append(
                SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=2, stride=1, padding=0,
                              pose_out=True))
            '''bn'''
            self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))

            self.conv_layers.append(
                SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1,
                              pose_out=True))
            '''bn'''
            self.norm_layers.append(nn.BatchNorm2d(caps_size * num_caps))

        '''恒等输出'''
        self.conv_a = nn.Conv2d(8 * planes, num_caps, kernel_size=3, stride=1, padding=1, bias=False)
        '''姿态变量'''
        self.conv_pose = nn.Conv2d(8 * planes, num_caps * caps_size, kernel_size=3, stride=1, padding=1, bias=False)
        '''两个bn'''
        self.bn_a = nn.BatchNorm2d(num_caps)
        self.bn_pose = nn.BatchNorm2d(num_caps * caps_size)
        # self.fc = SelfRouting2d(num_caps, classes, caps_size, 1, kernel_size=final_shape, padding=0, pose_out=False)

    def getCNN_sr(self):

        cnn = nn.Sequential(
            nn.Conv2d(1, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(True),
            nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(True),
            nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(True),
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # nn.BatchNorm2d(planes * 8),
            # nn.ReLU(True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes * 8),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # nn.BatchNorm2d(planes * 8),
            # nn.ReLU(True),
            # nn.Conv2d(planes * 8, planes * 8, kernel_size=2, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(planes * 8),
            # nn.ReLU(True),
        )

        return cnn

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
            BLSTM(64, 256, 256),
            BLSTM(256, 256, 256)
        )
        return rnn

    # image, length, text, text_rev, test
    def forward(self, input, text_length, text, text_rev, test=False):

        # input = self.stn(input)
        result, feature_pyramid = self.cnn(input)
        # (bs,512,1,5)
        # print('卷积层的输出结果', result.size())

        a, pose = self.conv_a(result), self.conv_pose(result)
        '''这里a需要sigmoid即归一化'''
        a, pose = torch.sigmoid(self.bn_a(a)), self.bn_pose(pose)

        # elif self.mode == 'SR':
        #     '''自回归模型'''
        #     self.conv_layers.append(SelfRouting2d(num_caps, num_caps, caps_size, caps_size, kernel_size=3, stride=1, padding=1, pose_out=True))
        #     '''bn'''
        #     self.norm_layers.append(nn.BatchNorm2d(planes*num_caps))

        '''深度就是depth，核心就在于这个SelfRouting2d函数了'''

        # print("pose的尺寸为",pose.size())
        # print("A的尺寸为", a.size())
        for m, bn in zip(self.conv_layers, self.norm_layers):
            a, pose = m(a, pose)
            pose = bn(pose)

        # print("在这里激活的尺寸为", a.size())  # 32 32 1 24
        # print("姿态的尺寸为",pose.size()) # 32,512,1,24

        '''这里忘记了改，白跑了一天'''
        result = a
        B, C, H, W = result.size()
        assert H == 1, 'The height of the input image must be 1.'
        result = result.squeeze(2)
        result = result.permute(2, 0, 1)

        result = self.rnn(result)
        '''feature, text_length, test sign'''
        # result = self.attention(result,text,text_length, test)
        result = self.attention(result, pose, feature_pyramid,text_length, text, test)
        return result


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128, CUDA=True):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        # caps_size = 6
        # num_caps = 64
        pose_dim = caps_size * num_caps
        pyramid_feature_dim = 576

        self.rnn = nn.GRUCell(input_size + num_embeddings + pose_dim + pyramid_feature_dim, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        # self.fracPickup = fracPickup(CUDA=CUDA)

        '''映射为x,y,w,h'''
        self.pose_linear = nn.Linear(input_size+pose_dim, 4)

    def forward(self, prev_hidden, feats, pose,pyramid, cur_embeddings, test=False):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size

        feats_proj = self.i2h(feats.view(-1, nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1, nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(
            -1, hidden_size)
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT, nB)

        alpha = F.softmax(emition, 0)  # nB * nT

        '''TODO：把姿态放进来'''
        # print("激活的尺寸为",a.size()) # 24,32,128
        # print("姿态的尺寸为",pose.size()) # 32,512,1,24

        '''先将feature特征合并'''


        pose = pose.squeeze(2)
        pose = pose.permute(2, 0, 1)  # 24,32,512
        feats = torch.cat([feats, pose], 2)
        nC = feats.size(2)

        pose_context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nPosdim
        coord = self.pose_linear(pose_context)
        coord = F.sigmoid(coord)  # nB * 4  (0,1)

        # coord = (coord + 1)

        crop_features = []
        for features in pyramid:
            # print("金字塔特征抽取的位置，特征尺寸为",features.size())

            '''在这个位置应该区分出不同尺寸特征图的坐标'''
            h, w  = features.size(2), features.size(3)
            # print("该特征图的宽，高为",h,w)

            scale = torch.Tensor([[h,w,h,w]]).cuda()
            coord = coord * scale

            crop_feature = roi_align(features,[coord],(2,2))
            # print("裁剪后的特征大小为",crop_feature.size())

            crop_feature = crop_feature.view(nB,-1)
            # print("裁剪后并放缩的特征大小为", crop_feature.size())
            # feats = torch.cat([feats, crop_feature], 2)
            crop_features.append(crop_feature)




        '''这里就不简单合并了，而是通过姿势向量，估计出一个坐标值'''
        # feats = torch.cat([feats,pose],2)


        '''TODO'''
        # 送入一个特征金字塔，然后用roi pooling / align  的方式去获取相应区域
        # 就假设CNN是可以学到坐标的吧


        # print("合并特征的尺寸为",feats.size())


        context = (feats * alpha.view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(0)  # nB * nC
        if len(context.size()) == 1:
            context = context.unsqueeze(0)
        context = torch.cat([context, cur_embeddings], 1)

        '''特征中加入金字塔'''
        for crop_feature in crop_features:
            context = torch.cat([context,crop_feature],1)

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
    def forward(self, feats, pose,pyramid, text_length, text, test=False):

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
                hidden, alpha = self.attention_cell(hidden, feats, pose,pyramid, cur_embeddings, test)
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
                hidden, alpha = self.attention_cell(hidden, feats, pose,pyramid, cur_embeddings, test)
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


class SelfRouting2d(nn.Module):
    def __init__(self, A, B, C, D, kernel_size=3, stride=1, padding=1, pose_out=False):
        super(SelfRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride

        self.pad = padding

        self.pose_out = pose_out

        if pose_out:
            self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B * D, C))
            nn.init.kaiming_uniform_(self.W1.data)

        self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1, self.kkA, B))

        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)

    def forward(self, a, pose):
        # a: [b, A, h, w]
        # pose: [b, AC, h, w]
        b, _, h, w = a.shape

        # [b, ACkk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, C, kk, l]
        pose = pose.view(b, self.A, self.C, self.kk, l)
        # [b, l, kk, A, C]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, C, 1]
        pose = pose.view(b, l, self.kkA, self.C, 1)

        if hasattr(self, 'W1'):
            # self.W1 = nn.Parameter(torch.FloatTensor(self.kkA, B * D, C))
            # [b, l, kkA, BD]
            pose_out = torch.matmul(self.W1, pose).squeeze(-1)
            # [b, l, kkA, B, D]
            pose_out = pose_out.view(b, l, self.kkA, self.B, self.D)

        # self.W2 = nn.Parameter(torch.FloatTensor(self.kkA, B, C))
        # [b, l, kkA, B]
        logit = torch.matmul(self.W2, pose).squeeze(-1) + self.b2

        '''可以想象成每个立方体的方块对于下一层的B个胶囊层的贡献度'''
        # [b, l, kkA, B]
        r = torch.softmax(logit, dim=3)  # c_ij?

        # [b, kkA, l]
        a = F.unfold(a, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a = a.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a = a.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA, 1]
        a = a.view(b, l, self.kkA, 1)

        # [b, l, kkA, B]
        ar = a * r
        # [b, l, 1, B]
        ar_sum = ar.sum(dim=2, keepdim=True)
        # [b, l, kkA, B, 1]
        coeff = (ar / (ar_sum)).unsqueeze(-1)

        # [b, l, B]
        # a_out = ar_sum.squeeze(2)
        a_out = ar_sum / a.sum(dim=2, keepdim=True)
        a_out = a_out.squeeze(2)

        # [b, B, l]
        a_out = a_out.transpose(1, 2)

        if hasattr(self, 'W1'):
            # [b, l, B, D]
            pose_out = (coeff * pose_out).sum(dim=2)
            # [b, l, BD]
            pose_out = pose_out.view(b, l, -1)
            # [b, BD, l]
            pose_out = pose_out.transpose(1, 2)

        # oh = ow = math.floor(l ** (1 / 2))

        oh = math.floor((h - self.k + 2 * self.pad) / self.stride + 1)
        ow = math.floor((w - self.k + 2 * self.pad) / self.stride + 1)

        # print(oh,ow)

        a_out = a_out.view(b, -1, oh, ow)
        if hasattr(self, 'W1'):
            pose_out = pose_out.view(b, -1, oh, ow)
        else:
            pose_out = None

        return a_out, pose_out


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
