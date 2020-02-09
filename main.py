# -*- coding:utf-8 -*-

'''
临时环境变量配置（今后废除）
'''

# 环境变量之后得去掉，模型进行重构，包括在所有from都写上明确的路径
import sys
sys.path.append('/home/cjy/FudanOCR/model/recognition_model/MORAN_V2')



'''
导入相关包
'''

# OCR架构中的包
from alphabet.alphabet import Alphabet
# from alphabet.ic13 import keys
from data import dataset
from config.yaml_reader import read_config_file
from model.recognition_model.MORAN_V2.models.moran import MORAN
from engine import trainer
from engine import tester
from utils import utils
from engine.trainer import Trainer
from engine.env import Env

# torch相关包
import os
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch
import argparse
import lmdb
import numpy as np
from numpy import random
from collections import OrderedDict

'''
===================================================================================
读取配置文件
===================================================================================
'''

# 读取配置文件
# parser = argparse.ArgumentParser()
# parser.add_argument('--config_file', required=True, help='path to config file')
# arg = parser.parse_args()
# # 通过一句话获得
# opt = read_config_file(arg.config_file)
# # 读取配置文件之后的一些工作，可以封装到其他地方去（杂活）
# cudnn.benchmark = True
# opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# np.random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# 创建一些文件夹（杂活）
# if opt.experiment is None:
#     opt.experiment = 'expr'
# os.system('mkdir {0}'.format(opt.experiment))

# 使用一句话完成上面的所有操作
env = Env()

'''
===================================================================================
字符表加载
===================================================================================
'''
# 检测是否是识别模型，是识别模型的话通过新建一个对象加载
alphabet = Alphabet(words = "0123456789")
nclass = len(alphabet)



# # 数据集，最后
# '''
# ===================================================================================
# 加载数据集
# ===================================================================================
# '''
#
# train_root = opt.train_nips
# val_root = opt.valroot
#
# train_dataset = dataset.lmdbDataset(root=opt.train_nips,
#     transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder,alphabet=alphabet)
# assert train_dataset
#
# test_dataset = dataset.lmdbDataset(root=opt.valroot,
#     transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder,alphabet=alphabet)
# assert test_dataset
#
# '''
# ===================================================================================
# 数据集预处理
# ===================================================================================
# '''
#
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=opt.batchSize,
#     shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.batchSize),
#     num_workers=int(opt.workers))
#
#
#
# '''
# ===================================================================================
# 导入模型
# ===================================================================================
# '''
#
# nc = 1
#
# converter = utils.strLabelConverterForAttention(alphabet)
# criterion = torch.nn.CrossEntropyLoss()
#
# if opt.cuda:
#     MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda)
# else:
#     MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder,
#                   inputDataType='torch.FloatTensor', CUDA=opt.cuda)
#
# # 预加载
# if opt.MORAN != '':
#     print('loading pretrained model from %s' % opt.MORAN)
#     if opt.cuda:
#         state_dict = torch.load(opt.MORAN)
#     else:
#         state_dict = torch.load(opt.MORAN, map_location='cpu')
#     MORAN_state_dict_rename = OrderedDict()
#     for k, v in state_dict.items():
#         name = k.replace("module.", "")  # remove `module.`
#         MORAN_state_dict_rename[name] = v
#     MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)
#
#
# # 优化器，可以为其写一个类
# if opt.adam:
#     optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# elif opt.adadelta:
#     optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
# elif opt.sgd:
#     optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
# else:
#     optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)
#
#
#
#
# '''
# ===================================================================================
# 训练
# ===================================================================================
# '''
#
# trainer = Trainer(model=MORAN,train_loader=train_loader,opt=opt,criterion=criterion,optimizer=optimizer,alphabet=alphabet,nc=nc)
# trainer.train()