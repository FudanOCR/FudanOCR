# -*- coding:utf-8 -*-
from engine.trainer import Trainer
from engine.env import Env
from data.build import build_dataloader

from engine.trainer_collection.MORAN import MORAN_Trainer
# from engine.trainer_collection.GRCNN import GRCNN_Trainer
from engine.trainer_collection.RARE import RARE_Trainer
# from engine.trainer_collection.CRNN import CRNN_Trainer
# from engine.trainer_collection.PixelLink import PixelLink_Trainer
# from engine.trainer_collection.LSN import LSN_Trainer
from engine.trainer_collection.AON import AON_Trainer



env = Env()
train_loader, test_loader = build_dataloader(env.opt)
newTrainer = MORAN_Trainer(modelObject=env.model, opt=env.opt, train_loader=train_loader,
                           val_loader=test_loader).train()

# import torch
# AON = env.model
# aon = AON(env.opt)
# input = torch.Tensor(1,1,100,100)
# # input = Variable(input)
# output = aon(input)
# print("Size: ",output.size())  # 1, 512, 1, 5

# import torch
# from component.convnet.resnet import getResNet18
#
# net = getResNet18()
# input = torch.Tensor(1,1,32,100)
# output = net(input)
# print(output.size())