import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision
from model.recognition_model.DAN import resnet

'''
Feature_Extractor
'''
class Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape):
        super(Feature_Extractor, self).__init__()
        self.model = resnet.resnet45(strides, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]