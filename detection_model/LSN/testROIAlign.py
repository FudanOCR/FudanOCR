from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchvision import models
import os
import torch.nn as nn
from torch.autograd import Variable
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
import _init_paths

base = torch.ones(1,1,100,100)
for i in range(100):
    for j in range(100):
        base[0][0][i][j]=i*100+j
print(base)
base = Variable(base.cuda())
rois = Variable(torch.FloatTensor([[0,0,0,6,6],[0,0,0,6,6]]).cuda())
print(rois)
roi_align = RoIAlignAvg(7,7,1.0/2)
roi = roi_align(base,rois)
print(roi)
# print(base)