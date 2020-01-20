"""
modify corresponding parameters according to yaml file
"""
from easydict import EasyDict
import torch
from yacs.config import CfgNode as CN

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 4

# training epoch number
config.max_epoch = 5

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.vis_num = 3

config.n_disk = 50      # 15

config.output_dir = 'output'

config.input_size = 512

config.dataset = 'ICDAR19'

config.device = 'cuda:1'

config.vis_dir = ''


def init_config(config, config_file):
    f = open(config_file)
    opt = CN.load_cfg(f)
    config.device = opt.device
    # config.vis_dir = opt.vis_dir
    config.out_dir = opt.out_dir
    # config.num_workers = opt.num_workers
    config.exp_name = opt.exp_name
    config.net = opt.net
    config.gpu_list = opt.gpu_list
    config.backbone = opt.backbone
    config.dataset = opt.dataset
    config.resume = opt.resume
    config.num_workers = opt.num_workers
    config.cuda = opt.cuda
    config.save_dir = opt.save_dir
    config.vis_dir = opt.vis_dir
    config.summary_dir = opt.summary_dir
    config.train_csv = opt.train_csv
    config.val_csv = opt.val_csv
    config.loss = opt.loss
    config.soft_ce = opt.soft_ce
    config.input_channel = opt.input_channel
    config.pretrain = opt.pretrain
    config.verbose = opt.verbose
    config.viz = opt.viz

    # train opts
    config.start_iter = opt.start_iter
    config.max_epoch = opt.max_epoch
    config.max_iters = opt.max_iters
    config.lr = opt.lr
    config.lr_adjust = opt.lr_adjust
    config.stepvalues = opt.stepvalues
    config.weight_decay = opt.weight_decay
    config.gamma = opt.gamma
    config.momentum = opt.momentum
    config.batch_size = opt.batch_size
    config.optim = opt.optim
    config.save_freq = opt.save_freq
    config.summary_freq = opt.summary_freq
    config.display_freq = opt.display_freq
    config.val_freq = opt.val_freq

    # data args
    config.rescale = opt.rescale
    config.means = opt.means
    config.stds = opt.stds
    config.input_size = opt.input_size

    # test args
    config.checkepoch = opt.checkepoch
    config.output_dir = opt.output_dir
    config.multi_scale = opt.multi_scale
    config.fixed = opt.fixed


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    if config.cuda == False:
        config.device = torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
