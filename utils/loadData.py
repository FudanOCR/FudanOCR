import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

def loadData(v, data):
    '''
    将数据data搬运到v中

    :param torch.Tensor v 目标张量
    :param torch.Tensor data 源张量
    '''
    try:
        v.resize_(data.size()).copy_(data)
    except:
        v.data.resize_(data.size()).copy_(data)