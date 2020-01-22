import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d as bn


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, use_bn=True, use_relu=True):
    layer = nn.Sequential()
    layer.add_module('conv', nn.Conv2d(in_channels,
                                       out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    if use_bn:
        layer.add_module('bn', nn.BatchNorm2d(out_channels))
    if use_relu:
        layer.add_module('relu', nn.ReLU())
    return layer



class denseASPP(nn.Module):
    def __init__(self):
        super(denseASPP, self).__init__()
        num_init_features_0 = 256
        num_init_features_1 = 256
        d_feature0 = 128  # model_cfg['d_feature0']
        d_feature1 = 64
        dropout0 = 0.1
        num_features = num_init_features_0
        num_features_2 = num_init_features_1

        self.block1 = nn.Sequential()
        self.block1.add_module('aspp3', _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False))
        self.block1.add_module('aspp6', _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0,
                                                        num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True))
        self.block1.add_module('aspp12', _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0,
                                                        num2=d_feature1,
                                                        dilation_rate=12, drop_out=dropout0, bn_start=True))
        self.block1.add_module('aspp18', _DenseAsppBlock(input_num=num_features + d_feature1 * 3, num1=d_feature0,
                                                        num2=d_feature1,
                                                        dilation_rate=18, drop_out=dropout0, bn_start=True))
        self.block1.add_module('smooth', conv_block(512, 256, kernel_size=1, padding=0, stride=1))

    def forward(self, x):
        # feature = x
        # for i in x:
        #     print(i.shape)

        feature_0 = x
        for layer in self.block1[:-1]:
            x_mid = layer(feature_0)
            # print(feature_0.shape, 'feature_0')
            feature_0 = torch.cat((x_mid, feature_0), dim=1)
        feature_0 = self.block1[-1](feature_0)



        return feature_0

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm_1', bn(input_num, momentum=0.0003)),

        self.add_module('relu_1', nn.ReLU()),
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm_2', bn(num1, momentum=0.0003)),
        self.add_module('relu_2', nn.ReLU()),
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature