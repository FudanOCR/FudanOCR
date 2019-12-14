import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict
from senet.se_module import SELayer
from torchsummary import summary

class DefaultCNN(nn.Module):
    def __init__(self, imgH, nc, leakyRelu=False):
        super(DefaultCNN, self).__init__()
        assert imgH % 16 == 0, 'Image height has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
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

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5, True)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.cnn = cnn
        print("Initializing cnn net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        conv = self.cnn(input)
        return conv


def defaultcnn(**kwargs):
    model = DefaultCNN(imgH=32, nc=1)
    return model


# -------------------------------------------------------------------------------#

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, iblock):
        super(_Transition, self).__init__()
        assert iblock < 4, "There are maximal 4 blocks."
        self.ks = [2, 2, 2]
        self.h_ss = [2, 2, 2]
        self.w_ss = [1, 1, 1]
        self.w_pad = [1, 1, 1]
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d((self.ks[iblock], self.ks[iblock]),
                                             (self.h_ss[iblock], self.w_ss[iblock]),
                                             (0, self.w_pad[iblock])))


class DenseNet(nn.Module):
    def __init__(self, num_in, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_in, num_init_features, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        num_features = num_init_features

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, iblock=i)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Official init from torch repo
        # print("Initializing Dense net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = self.relu(features)
        # out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)

        return out


def DenseNet121(**kwargs):
    print("Initializing DenseNet121 net weights...")
    model = DenseNet(num_in=1, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    return model


def DenseNet169(**kwargs):
    print("Initializing DenseNet169 net weights...")
    model = DenseNet(num_in=1, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model


def DenseNet201(**kwargs):
    print("Initializing DenseNet201 net weights...")
    model = DenseNet(num_in=1, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    return model


#######Resnet#######

def conv3x3(in_planes, out_planes, stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_in, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=(2, 2))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 1))

        # Official init from torch repo
        # print("Initializing Resnet net weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def ResNet18(**kwargs):
    print("Initializing Resnet18 net weights...")
    model = ResNet(num_in=1, block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return model


def ResNet34(**kwargs):
    print("Initializing Resnet34 net weights...")
    model = ResNet(num_in=1, block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return model


def ResNet50(**kwargs):
    print("Initializing Resnet50 net weights...")
    model = ResNet(num_in=1, block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    return model


def ResNet101(**kwargs):
    model = ResNet(num_in=1, block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)
    return model


def ResNet101(**kwargs):
    model = ResNet(num_in=1, block=Bottleneck, layers=[3, 4, 36, 3], **kwargs)
    return model


#######-----------SEResnet-----------#######
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet(nn.Module):

    def __init__(self, num_in, block, layers):
        self.inplanes = 64
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=(2, 2))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 1))

        # Official init from torch repo
        # print("Initializing SEResNet weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=(1, 1)):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def se_resnet18(**kwargs):
    print("Initializing SE_Resnet18 net weights...")
    model = SEResNet(num_in=1, block=SEBasicBlock, layers=[2, 2, 2, 2], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(**kwargs):
    print("Initializing SE_Resnet34 net weights...")
    model = SEResNet(num_in=1, block=SEBasicBlock, layers=[3, 4, 6, 3], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(**kwargs):
    print("Initializing SE_Resnet50 net weights...")
    model = SEResNet(num_in=1, block=SEBottleneck, layers=[3, 4, 6, 3], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(**kwargs):
    model = SEResNet(num_in=3, block=SEBottleneck, layers=[3, 4, 23, 3], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(**kwargs):
    model = SEResNet(num_in=3, block=SEBottleneck, layers=[3, 8, 36, 3], **kwargs)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


#######-----------SEcnn-----------#######
class selayer(nn.Module):
    def __init__(self, i):
        super(selayer, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]

        self.conv = nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i])
        self.bn = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(nOut, 16)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)

        return out


class SECNN(nn.Module):
    def __init__(self, imgH, nc, leakyRelu=False):
        super(SECNN, self).__init__()
        assert imgH % 16 == 0, 'Image height has to be a multiple of 16'

        self.layer0 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.layer1 = selayer(1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = selayer(2)
        self.layer3 = selayer(3)
        self.pool2 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.layer4 = selayer(4)
        self.layer5 = selayer(5)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.layer6 = selayer(6)
        # self.conv6 = nn.Conv2d(512, 512, 2, 1, 0)
        # self.bn6 = nn.BatchNorm2d(512)

        print("Initializing secnn weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        # print('input:', input.shape)
        x = self.layer0(input)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool3(x)
        x = self.layer6(x)
        # x = self.conv6(x)
        # x = self.bn6(x)
        # print ('secnn_out:',x.shape)
        return x


def secnn(**kwargs):
    model = SECNN(imgH=32, nc=1)
    # print(model)
    return model
if __name__ == '__main__':
    model = secnn().cuda()
    summary(model, (3, 32, 100))
