import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """
    '''
    一个用于对torch.Variable或者torch.Tensor求平均值的类
    '''

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res