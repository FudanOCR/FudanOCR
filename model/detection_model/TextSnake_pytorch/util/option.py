"""
modify corresponding parameters according to yaml file
"""
import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime

from yacs.config import CfgNode as CN

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str

class BaseOptions(object):

    def __init__(self, config_file):
        f = open(config_file)
        opt = CN.load_cfg(f)

        # basic opts
        self.exp_name = opt.exp_name
        self.net = opt.net
        self.gpu_list = opt.gpu_list
        self.backbone = opt.backbone
        self.dataset = opt.dataset
        self.resume = opt.resume
        self.num_workers = opt.num_workers
        self.cuda = opt.cuda
        self.save_dir = opt.save_dir
        self.vis_dir = opt.vis_dir
        self.summary_dir = opt.summary_dir
        self.train_csv = opt.train_csv
        self.val_csv = opt.val_csv
        self.loss = opt.loss
        self.soft_ce = opt.soft_ce
        self.input_channel = opt.input_channel
        self.pretrain = opt.pretrain
        self.verbose = opt.verbose
        self.viz = opt.viz

        # train opts
        self.start_iter = opt.start_iter
        self.max_epoch = opt.max_epoch
        self.max_iters = opt.max_iters
        self.lr = opt.lr
        self.lr_adjust = opt.lr_adjust
        self.stepvalues = opt.stepvalues
        self.weight_decay = opt.weight_decay
        self.gamma = opt.gamma
        self.momentum = opt.momentum
        self.batch_size = opt.batch_size
        self.optim = opt.optim
        self.save_freq = opt.save_freq
        self.summary_freq = opt.summary_freq
        self.display_freq = opt.display_freq
        self.val_freq = opt.val_freq

        # data args
        self.rescale = opt.rescale
        self.means = opt.means
        self.stds = opt.stds
        self.input_size = opt.input_size

        # test args
        self.checkepoch = opt.checkepoch
        self.output_dir = opt.output_dir
        self.multi_scale = opt.multi_scale
        self.fixed = opt.fixed

    def initialize(self):

        # Setting default torch Tensor type
        if self.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.save_dir, self.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self

