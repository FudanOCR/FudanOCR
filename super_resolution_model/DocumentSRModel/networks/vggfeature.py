import torch
import torch.nn as nn


class VGGFeatureMap(nn.Module):
    def __init__(self, netVGG, feature_layer=7):
        super(VGGFeatureMap, self).__init__()
        self.featuremap = nn.Sequential(*list(netVGG.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.featuremap(x)