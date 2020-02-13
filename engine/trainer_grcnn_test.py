import time
from utils import utils
import torch
from torch.autograd import Variable
import os
import Levenshtein

from engine.loss import getLoss
from engine.optimizer import getOptimizer
from alphabet.alphabet import Alphabet


class Trainer(object):

    def __init__(self, modelObject, opt='', train_loader='', val_loader=''):
        '''
        模型训练器主体

        传入参数：
        1.未初始化的模型modelObject
        2.解析好的参数opt
        3.训练数据集的加载器train_loader
        4.验证数据集的加载器val_loader

        TODO：
        1.根据参数文件实例化模型
        2.配置Loss与Optimizer
        '''

        # 基本信息
        self.opt = opt
        # 读取识别组件需要的字符表
        self.alphabet = Alphabet(self.opt.ADDRESS.RECOGNITION.ALPHABET)

        '''初始化模型，并且转载预训练模型'''
        self.model = self.initModel(modelObject)
        self.loadParam()

        self.optimizer = getOptimizer(self.model, self.opt)
        self.criterion = getLoss(self.opt)

        self.train_loader = train_loader
        self.val_loader = val_loader

        '''识别模型工具元件'''
        self.loss_avg = utils.averager()
        self.converter = utils.strLabelConverterForAttention(self.alphabet.str)

        '''常量区'''
        self.i = 0
        self.highestAcc = 0

    def initModel(self, modelObject):
        '''
        根据配置文件初始化模型
        '''
        if self.opt.CUDA:
            return modelObject(self.opt, self.alphabet).cuda()
        else:
            modelObject(self.opt, self.alphabet)

    def loadParam(self):
        '''
        预加载训练参数
        '''
        if os.path.isfile(self.opt.ADDRESS.PRETRAIN_MODEL_DIR):

            address = self.opt.ADDRESS.PRETRAIN_MODEL_DIR
            print('loading pretrained model from %s' % address)
            if opt.CUDA:
                state_dict = torch.load(address)
            else:
                state_dict = torch.load(address, map_location='cpu')
            state_dict_rename = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")  # remove `module.`
                state_dict_rename[name] = v
            self.model.load_state_dict(state_dict_rename, strict=True)

    def pretreatment(self, data):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据
        '''
        cpu_images, cpu_gt = data
        v_images = Variable(cpu_images.cuda())
        return (v_images)


    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''

        if test == False:
            cpu_images, cpu_gt = originData
            text, text_len = converter.encode(cpu_gt)
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)
            bsz = cpu_images.size(0)
            predict_len = Variable(torch.IntTensor([modelResult.size(0)] * bsz))
            cost = criterion(modelResult, v_gt, predict_len, v_gt_len)
            return cost

        else:
            cpu_images, cpu_gt = originData
            bsz = cpu_images.size(0)
            text, text_len = converter.encode(cpu_gt)
            v_Images = Variable(cpu_images.cuda())
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)

            predict = model(v_Images)
            predict_len = Variable(torch.IntTensor([predict.size(0)] * bsz))
            cost = criterion(predict, v_gt, predict_len, v_gt_len)

            sim_preds = converter.decode(acc.data, predict_len.data, raw=False)

            return cost, sim_preds, cpu_gt

    def validate(self):
        '''
        在特定训练次数后执行验证模型能力操作
        '''

        acc_tmp = 0
        self.setModelState('test')

        print('Start val')
        val_loader = self.val_loader
        val_iter = iter(val_loader)
        n_correct = 0
        n_total = 0
        distance = 0.0
        loss_avg = self.loss_avg

        f = open('./OCR新架构验证测试.txt', 'a', encoding='utf-8')

        for i in range(len(val_loader)):
            data = val_iter.next()

            pretreatmentData = self.pretreatment(data)

            modelResult = self.model(*pretreatmentData)

            cost, preds, targets = self.posttreatment(modelResult, pretreatmentData, originData=data, test=True)

            loss_avg.add(cost)

            for pred, target in zip(preds, targets):
                if pred == target.lower():
                    n_correct += 1
                f.write("预测 %s      目标 %s\n" % (pred, target))
                distance += Levenshtein.distance(pred, target) / max(len(pred), len(target))
                n_total += 1

        f.close()
        accuracy = n_correct / float(n_total)

        print("correct / total: %d / %d, " % (n_correct, n_total))
        print('levenshtein distance: %f' % (distance / n_total))
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

        if acc_tmp > self.highestAcc:
            self.highestAcc = acc_tmp
            torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                self.opt.ADDRESS.CHECKPOINTS_DIR, self.i, str(self.highestAcc)[:6]))
        return acc_tmp

    def train(self):
        '''
        训练函数
        Already Done:
        1.初始化一个可训练模型
        2.定义损失函数与优化器

        TODO:
        1.定义一个epoch循环，循环总数在配置文件中定义。每个epoch会把所有
        2.定义一个train_loader迭代器，在每一个iter里，使用next()获取下一个批次
            对于每一个iter,计算模数，在对应的迭代周期里执行保存/验证/记录 操作
        '''

        t0 = time.time()
        self.highestAcc = 0
        for epoch in range(self.opt.MODEL.EPOCH):

            self.i = 0
            train_iter = iter(self.train_loader)

            while self.i < len(self.train_loader):

                '''检查该迭代周期是否需要保存或验证'''
                self.checkSaveOrVal()

                data = train_iter.next()

                pretreatmentData = self.pretreatment(data)

                modelResult = self.model(*pretreatmentData)

                cost = self.posttreatment(modelResult, pretreatmentData, data)

                self.model.zero_grad()
                cost.backward()
                self.optimizer.step()

                self.loss_avg.add(cost)

                if self.i % self.opt.SHOW_FREQ == 0:
                    t1 = time.time()
                    print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                          (epoch, self.opt.MODEL.EPOCH, self.i, len(self.train_loader), self.loss_avg.val(), t1 - t0)),
                    self.loss_avg.reset()
                    t0 = time.time()

                self.i += 1

    def checkSaveOrVal(self):
        '''验证'''
        if self.i % self.opt.VAL_FREQ == 0:
            self.setModelState('test')
            acc_tmp = self.validate()
            '''记录训练结果最大值的模型文件'''
            if acc_tmp > self.highestAcc:
                self.highestAcc = acc_tmp
                torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                    self.opt.ADDRESS.CHECKPOINTS_DIR, self.i, str(self.highestAcc)[:6]))

        '''保存'''
        if self.i % self.opt.SAVE_FREQ == 0:
            torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                self.opt.ADDRESS.CHECKPOINTS_DIR, self.opt.MODEL.EPOCH, self.i))

        '''恢复训练状态'''
        self.setModelState('train')

    def setModelState(self,state):
        '''
        根据传入的状态判断模型处于训练或者验证状态
        '''
        if state == 'eval' or state == 'test':
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        elif state == 'train':
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()


