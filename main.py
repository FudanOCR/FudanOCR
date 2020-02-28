# -*- coding:utf-8 -*-
from engine.trainer import Trainer
from engine.env import Env
from data.build import build_dataloader


class GRCNN_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data, test=False):
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

    def getScheduler(self):
        '''动态调整lr'''
        from torch.optim.lr_scheduler import LambdaLR, StepLR
        # return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        return StepLR(self.optimizer, step_size=20, gamma=0.1)


class TextSnake_Trainer(Trainer):

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = data

        return img

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        from model.detection_model.TextSnake_pytorch.util.misc import to_device

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = originData
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)
        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            self.criterion(modelResult, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask, total_iter)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

        return loss


class MORAN_Trainer(Trainer):
    '''
    重载训练器

    主要重载函数为pretreatment与posttreatment两个函数，为数据前处理与数据后处理
    数据前处理：从dataloader加载出来的数据需要经过加工才可以进入model进行训练
    数据后处理：经过模型训练的数据是编码好的，需要进一步解码用来计算损失函数
    不同模型的数据前处理与后处理的方式不一致，因此需要进行函数重载
    '''

    def __init__(self, modelObject, opt, train_loader, val_loader):
        Trainer.__init__(self, modelObject, opt, train_loader, val_loader)

    def pretreatment(self, data, test=False):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据
        '''
        import torch
        from torch.autograd import Variable
        from utils import utils

        image = torch.FloatTensor(self.opt.MODEL.BATCH_SIZE, self.opt.IMAGE.IMG_CHANNEL, self.opt.IMAGE.IMG_H,
                                  self.opt.IMAGE.IMG_H)
        text = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        text_rev = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        length = torch.IntTensor(self.opt.MODEL.BATCH_SIZE)

        if self.opt.CUDA:
            # self.model = torch.nn.DataParallel(self.model, device_ids=range(self.opt.ngpu))
            image = image.cuda()
            text = text.cuda()
            text_rev = text_rev.cuda()
            self.criterion = self.criterion.cuda()

        image = Variable(image)
        text = Variable(text)
        text_rev = Variable(text_rev)
        length = Variable(length)

        if self.opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            t_rev, _ = self.converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            return image, length, text, text_rev, test
            # preds0, preds1 = self.model(image, length, text, text_rev)
            # cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            return image, length, text, text_rev, test
            # preds = self.model(image, length, text, text_rev)
            # cost = self.criterion(preds, text)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''
        import torch
        from torch.autograd import Variable
        from utils import utils

        if test == False:
            if self.opt.BidirDecoder:
                image, length, text, text_rev, _ = pretreatmentData
                preds0, preds1 = modelResult
                cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
            else:
                image, length, text, text_rev = pretreatmentData
                preds = self.model(image, length, text, text_rev)
                cost = self.criterion(preds, text)

            return cost

        else:
            if self.opt.BidirDecoder:
                preds0, preds1 = modelResult
                cpu_images, cpu_texts, cpu_texts_rev = originData
                image, length, text, text_rev, _ = pretreatmentData

                cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
                preds0, preds1 = modelResult
                preds0_prob, preds0 = preds0.max(1)
                preds0 = preds0.view(-1)
                preds0_prob = preds0_prob.view(-1)
                sim_preds0 = self.converter.decode(preds0.data, length.data)
                preds1_prob, preds1 = preds1.max(1)
                preds1 = preds1.view(-1)
                preds1_prob = preds1_prob.view(-1)
                sim_preds1 = self.converter.decode(preds1.data, length.data)
                sim_preds = []
                for j in range(cpu_images.size(0)):
                    text_begin = 0 if j == 0 else length.data[:j].sum()
                    if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).data[0] > \
                            torch.mean(
                                preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).data[0]:
                        sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                    else:
                        sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')

                return cost, sim_preds, cpu_texts
            else:
                cpu_images, cpu_texts = originData
                preds = modelResult
                image, length, text, text_rev, _ = pretreatmentData
                cost = self.criterion(preds, text)
                _, preds = preds.max(1)
                preds = preds.view(-1)
                sim_preds = self.converter.decode(preds.data, length.data)

                return cost, sim_preds, cpu_texts

    def getScheduler(self):
        from torch.optim.lr_scheduler import LambdaLR, StepLR
        return LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))


env = Env()
train_loader, test_loader = build_dataloader(env.opt)
newTrainer = GRCNN_Trainer(modelObject=env.model, opt=env.opt, train_loader=train_loader, val_loader=test_loader).train()
