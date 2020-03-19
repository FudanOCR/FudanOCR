import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class tcm(nn.Module):
    """docstring for TCM"""
    def __init__(self,out_channels):
        super(tcm, self).__init__()
        self.conv0=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.convvvv1= nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bnnnn1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.convvvv2 = nn.Conv2d(out_channels, 2, kernel_size=1)
#         self.bnnnn2 = nn.BatchNorm2d(2, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=2)
#         for l in [self.convvvv1, self.convvvv2]:
#             torch.nn.init.normal_(l.weight, std=0.01)
#             torch.nn.init.constant_(l.bias, 0)
            
        for module in [self.conv0,self.convvvv1, self.convvvv2]:
                # Caffe2 implementation uses XavierFill, which in fact
                # corresponds to kaiming_uniform_ in PyTorch
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)
    def forward(self,x):
        att=self.conv0(x)
        mid=self.relu(self.bnnnn1(self.convvvv1(att)))
        map=self.relu(self.convvvv2(mid))
        saliency_map=torch.exp(self.softmax(map))[:,:1,:,:]
        saliency_map = saliency_map.expand_as(x)

        return saliency_map
 