# -*- coding:utf-8 -*-

from model.recognition_model.GRCNN.models.crann import newCRANN
'''MORAN'''
# from engine.trainer import Trainer
'''GRCNN'''
from engine.trainer_grcnn_test import Trainer
from engine.env import Env
from data.getdataloader import getDataLoader


env = Env()
train_loader , test_loader = getDataLoader(env.opt)
trainer = Trainer(modelObject=newCRANN, opt=env.opt, train_loader=train_loader,val_loader=test_loader).train()
