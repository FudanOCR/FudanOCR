import os
import torch
import datetime
import numpy as np
from tqdm import tqdm

import model.detection_model.AdvancedEAST.config as cfg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_log(losses, epoch, current_batch, loader_len, tock, split='Training'):
    '''Save log of losses.'''
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
    log_path = os.path.join(cfg.result_dir, tock + '-log.txt')

    with open(log_path, 'a') as f:
        line = 'Epoch: [{0}][{1}/{2}] {3} Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
            epoch + 1, current_batch, loader_len, split, loss=losses)
        f.write(line)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'init_type [%s] not implemented.' % init_type)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
