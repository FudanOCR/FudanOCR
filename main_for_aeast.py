# -*- coding:utf-8 -*-
# from model.recognition_model.GRCNN.models.crann import newCRANN
# from model.recognition_model.MORAN_V2.models.moran import newMORAN
from model.detection_model.AdvancedEAST.network.AEast import East
from engine.trainer_AEAST import Trainer
from engine.env import Env
from data.getdataloader_aeast import getDataLoader

class AEAST_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据
        '''
        from torch.autograd import Variable
        cpu_images, cpu_gt = data
        v_images = Variable(cpu_images.cuda())
        return (v_images,)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''
        from torch.autograd import Variable
        import torch
        if test == False:
            cpu_images, cpu_gt = originData
            text, text_len = self.converter.encode(cpu_gt)
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)
            bsz = cpu_images.size(0)
            predict_len = Variable(torch.IntTensor([modelResult.size(0)] * bsz))
            cost = self.criterion(modelResult, v_gt, predict_len, v_gt_len)
            return cost

        else:
            cpu_images, cpu_gt = originData
            bsz = cpu_images.size(0)
            text, text_len = self.converter.encode(cpu_gt)
            v_Images = Variable(cpu_images.cuda())
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)

            predict = modelResult
            # modelResult = self.model(v_Images)
            predict_len = Variable(torch.IntTensor([modelResult.size(0)] * bsz))
            cost = self.criterion(predict, v_gt, predict_len, v_gt_len)

            _, acc = predict.max(2)
            acc = acc.transpose(1, 0).contiguous().view(-1)

            sim_preds = self.converter.decode(acc.data, predict_len.data)

            return cost, sim_preds, cpu_gt


env = Env()
train_loader, test_loader = getDataLoader(env.opt)
newTrainer = AEAST_Trainer(modelObject=East, opt=env.opt, train_loader=train_loader, val_loader=test_loader).train()
