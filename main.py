# -*- coding:utf-8 -*-

'''
临时环境变量配置（今后废除）
'''

# 环境变量之后得去掉，模型进行重构，包括在所有from都写上明确的路径
import sys
from utils import utils

sys.path.append('.')
sys.path.append('/home/cjy/FudanOCR/model/recognition_model/MORAN_V2')
sys.path.append('/home/cjy/FudanOCR/model/recognition_model/GRCNN')
sys.path.append('/home/cjy/FudanOCR/model/recognition_model')
'''
导入相关包
'''
# OCR架构中的包
from alphabet.alphabet import Alphabet
from data import dataset
from model.recognition_model.MORAN_V2.models.moran import newMORAN
from model.recognition_model.GRCNN.models.crann import newCRANN

from engine.trainer import Trainer
from engine.env import Env
from logger.logger import Logger

# torch相关包
import os
from torch.autograd import Variable
import torch
import lmdb
import json

'''
===================================================================================
读取配置文件
===================================================================================
'''
env = Env()
opt = env.getOpt()




'''
===================================================================================
字符表加载
===================================================================================
'''
# 检测是否是识别模型，是识别模型的话通过新建一个对象加载
alphabet = Alphabet(env.opt.ADDRESS.RECOGNITION.ALPHABET)

'''
===================================================================================
加载数据集
===================================================================================
'''
train_root = opt.ADDRESS.RECOGNITION.TRAIN_DATA_DIR
val_root = opt.ADDRESS.RECOGNITION.TEST_DATA_DIR

train_dataset = dataset.lmdbDataset(root=train_root,
                                    transform=dataset.resizeNormalize((opt.IMAGE.IMG_W, opt.IMAGE.IMG_H)),
                                    reverse=opt.BidirDecoder, alphabet=alphabet.str)
assert train_dataset

test_dataset = dataset.lmdbDataset(root=val_root,
                                   transform=dataset.resizeNormalize((opt.IMAGE.IMG_W, opt.IMAGE.IMG_H)),
                                   reverse=opt.BidirDecoder, alphabet=alphabet.str)
assert test_dataset

'''
===================================================================================
数据集预处理
===================================================================================
'''
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.MODEL.BATCH_SIZE,
    shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.MODEL.BATCH_SIZE),
    num_workers=int(opt.BASE.WORKERS))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.MODEL.BATCH_SIZE,
    shuffle=False,
    num_workers=int(opt.BASE.WORKERS))

'''
===================================================================================
训练
===================================================================================
'''
trainer = Trainer(modelObject=newCRANN, opt=opt, train_loader=train_loader,val_loader=test_loader)
trainer.train()
