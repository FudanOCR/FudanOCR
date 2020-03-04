import time
import torch
from torch.autograd import Variable
import os
import Levenshtein
import shutil
import urllib.request as request
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR, StepLR
from tqdm import tqdm
import numpy as np
import json

from engine.loss import getLoss
from engine.optimizer import getOptimizer
from alphabet.alphabet import Alphabet
from engine.pretrain import pretrain_model
from logger.info import file_summary
from logger.logger import Logger
from utils.average import averager
from utils.Pascal_VOC import eval_func
from utils.AverageMeter import AverageMeter
from utils.downloadCallback import callbackfunc



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

        '''初始化模型，并且加载预训练模型'''
        self.model = self.initModel(modelObject)
        self.loadParam()
        self.loadTool()

        self.optimizer = getOptimizer(self.model, self.opt)
        self.criterion = getLoss(self.opt)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.Logger = Logger(self.opt.VISUALIZE.TAG)

        '''动态调整lr'''
        self.scheduler = None
        if self.opt.MODEL.DYNAMIC_LR == True:
            self.scheduler = self.getScheduler()
            assert self.scheduler, "opt.MODEL.DYNAMIC_LR == True. You need to overload the function getScheduler(), then return a valid scheduler."

        '''设置finetune'''
        if self.opt.FUNCTION.FINETUNE == True:
            assert self.finetune(), "opt.FUNCTION.FINETUNE == True. You need to overload the function finetune() to adjust the model or the optimizer."



    def initModel(self, modelObject):
        '''
        根据配置文件初始化模型
        '''
        if self.opt.BASE.CUDA:
            return modelObject(self.opt).cuda()
        else:
            modelObject(self.opt)

    def loadTool(self):
        '''
        根据模型的类型，加载相应的组件
        '''
        if self.opt.BASE.TYPE == 'R':
            self.alphabet = Alphabet(self.opt.ADDRESS.ALPHABET)
            if self.opt.BASE.MODEL == 'MORAN':
                from utils.strLabelConverterForAttention import strLabelConverterForAttention
                self.converter = strLabelConverterForAttention(self.alphabet.str)
            elif self.opt.BASE.MODEL == 'GRCNN' or self.opt.BASE.MODEL == 'CRNN':
                from utils.strLabelConverterForCTC import strLabelConverterForCTC
                self.converter = strLabelConverterForCTC(self.alphabet.str)
            self.highestAcc = 0

            self.val_times = 0


    def loadParam(self):
        '''
        预加载训练参数，分为以下三种情况
        1.空目录：不作处理
        2.本地文件：加载本地预训练模型
        3.本地目录：下载云端预训练模型至该目录下
        '''
        if self.opt.ADDRESS.PRETRAIN_MODEL_DIR == '':
            '''空字符串'''
            return

        elif os.path.isfile(self.opt.ADDRESS.PRETRAIN_MODEL_DIR):
            '''本地文件'''
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

        else:
            '''
            本地目录，如果存在则删除原有目录再安装一个新目录
            '''
            if os.path.exists(self.opt.ADDRESS.PRETRAIN_MODEL_DIR):
                shutil.rmtree(self.opt.ADDRESS.PRETRAIN_MODEL_DIR)
            os.makedirs(self.opt.ADDRESS.PRETRAIN_MODEL_DIR)

            model_name = self.opt.BASE.MODEL
            print('Load pretrained model from : ', pretrain_model[model_name])



            request.urlretrieve(pretrain_model[model_name], os.path.join(self.opt.ADDRESS.PRETRAIN_MODEL_DIR,
                                                                                pretrain_model[model_name].split('/')[
                                                                                    -1]),callbackfunc)
            print("Finish loading!")

            address = os.path.join(self.opt.ADDRESS.PRETRAIN_MODEL_DIR, pretrain_model[model_name].split('/')[-1])
            if self.opt.CUDA:
                state_dict = torch.load(address)
            else:
                state_dict = torch.load(address, map_location='cpu')
            state_dict_rename = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")  # remove `module.`
                state_dict_rename[name] = v
            self.model.load_state_dict(state_dict_rename, strict=True)

    def pretreatment(self, data, test=False):
        '''
        将从dataloader加载出来的data转化为可以传入神经网络的数据

        传入数据：
        data：从dataloader中反复迭代得到的数据，模型应该返回可以被神经网络接受的数据
        返回数据：
        数据长度为1，则返回形式为(returndata, )
        数据长度大于1，则返回形式为(returndata1, returndata2 ,returndata3 )
        '''
        pass

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据

        传入数据：
        modelResult:模型传出的结果
        pretreatmentData:能够被模型所接受的数据，即pretreatment函数返回的数据
        originData:从dataloader迭代出来的数据

        返回数据：
            训练阶段：
                返回cost
            验证阶段：
                返回cost，和其他可以用于评价模型能力的指标
                对于识别模型来说：其他可以用于评价模型能力的指标包括：target label 与 predict label
                对于检测模型来说：TODO
        '''
        pass

    def validate(self, epoch, iteration):
        '''
        将验证函数拆分为识别和检测两部分
        '''
        if self.opt.BASE.TYPE == 'R':
            return self.validate_recognition(epoch, iteration)
        elif self.opt.BASE.TYPE == 'D':
            return self.validate_detection(epoch, iteration)

    def validate_recognition(self, epoch, iteration):
        '''
        在特定训练次数后执行验证模型能力操作
        '''

        print('Start val')
        val_loader = self.val_loader
        val_iter = iter(val_loader)
        n_correct = 0
        n_total = 0
        distance = 0.0
        loss_avg = averager()

        self.val_times += 1

        for i in range(len(val_loader)):
            data = val_iter.next()

            pretreatmentData = self.pretreatment(data, True)

            modelResult = self.model(*pretreatmentData)

            cost, preds, targets = self.posttreatment(modelResult, pretreatmentData, originData=data, test=True)

            loss_avg.add(cost)

            for pred, target in zip(preds, targets):
                if pred.lower() == target.lower():
                    n_correct += 1

                '''利用logger工具将结果记录于文件夹中'''
                file_summary(self.opt.ADDRESS.LOGGER_DIR,str(self.val_times) +'_'+self.opt.BASE.MODEL+ "_result" +".txt","预测 %s      目标 %s\n" % (pred, target))

                '''添加功能：对正确文本和错误文本进行分类'''
                if pred.lower() == target.lower():
                    file_summary(self.opt.ADDRESS.LOGGER_DIR,str(self.val_times) +'_'+self.opt.BASE.MODEL+ "_right"+".txt","预测 %s      目标 %s\n" % (pred, target))
                else:
                    file_summary(self.opt.ADDRESS.LOGGER_DIR,str(self.val_times) +'_'+self.opt.BASE.MODEL+ "_wrong"+".txt","预测 %s      目标 %s\n" % (pred, target))

                distance += Levenshtein.distance(pred.lower(), target.lower()) / max(len(pred), len(target))
                n_total += 1

        accuracy = n_correct / float(n_total)
        '''利用logger工具将结果进行可视化'''
        total_index = epoch*(len(self.train_loader)//self.opt.FREQ.VAL_FREQ+1)+iteration//self.opt.FREQ.VAL_FREQ
        self.Logger.scalar_summary('Levenshtein Distance', distance / n_total, total_index)
        self.Logger.scalar_summary('Accuracy', accuracy, total_index)
        self.Logger.scalar_summary('Avg Loss', loss_avg.val(), total_index)

        print("correct / total: %d / %d, " % (n_correct, n_total))
        print('levenshtein distance: %f' % (distance / n_total))
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

        return accuracy

    def validate_detection(self, epoch, iteration):
        print('Start val')
        val_loader = self.val_loader
        val_iter = iter(val_loader)
        input_json_path = self.res2json()
        gt_json_path = self.opt.ADDRESS.GT_JSON_DIR

        # loss
        losses = AverageMeter()
        for i in range(len(val_loader)):
            data = val_iter.next()
            pretreatmentData = self.pretreatment(data)
            img = self.get_img_data(pretreatmentData)
            modelResult = self.model(img)
            loss = self.posttreatment(modelResult, pretreatmentData, data, True)
            print("No.%d, loss:%f" % (i, loss))
            file_summary(self.opt.ADDRESS.LOGGER_DIR, self.opt.BASE.MODEL + "_result.txt",
                         "No.%d, loss:%f \n" % (i, loss))
            losses.update(loss.item(), img.size(0))
        tqdm.write('Validate Loss - Avg Loss {0}'.format(losses.avg))

        # Precision / Recall / F_score
        precision, recall, f_score = \
            eval_func(input_json_path, gt_json_path, self.opt)

        # Generate log
        total_index = epoch*(len(self.train_loader)//self.opt.FREQ.VAL_FREQ+1)+iteration//self.opt.FREQ.VAL_FREQ
        self.Logger.scalar_summary('Precision', precision, total_index)
        self.Logger.scalar_summary('Recall', recall, total_index)
        self.Logger.scalar_summary('F_score', f_score, total_index)
        self.Logger.scalar_summary('Avg Loss', losses.avg, total_index)

        return precision

    def get_img_data(self, pretreatmentData):
        '''
        从pretreatment中提取出img数据，可根据需要重载
        '''
        return pretreatmentData[0]

    def res2json(self):
        '''
        生成res.json文件，可根据需要进行重载
        :return:
        '''
        result_dir = self.opt.ADDRESS.RESULT_DIR
        res_list = os.listdir(result_dir)
        res_dict = {}
        for rf in tqdm(res_list, desc='toJSON'):
            if rf[-4:] == '.txt':
                respath = os.path.join(result_dir, rf)
                with open(respath, 'r') as f:
                    reslines = f.readlines()
                reskey = rf[:-4]
                res_dict[reskey] = [{'points': np.rint(np.asarray(l.replace('\n', '').split(','), np.float32)).astype(
                    np.int32).reshape(-1, 2).tolist()} for l in reslines]

        jpath = os.path.join(result_dir, 'res.json')
        with open(jpath, 'w') as jf:
            json.dump(res_dict, jf)
        return jpath

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

        '''如果只需验证，只需要一次验证过程即可无需训练'''
        if self.opt.FUNCTION.VAL_ONLY == True:
            self.validate(1, self.opt.FREQ.VAL_FREQ)
            return

        loss_avg = averager()

        t0 = time.time()
        self.highestAcc = 0
        iteration = 0

        for epoch in range(self.opt.MODEL.EPOCH):

            train_iter = iter(self.train_loader)

            while iteration < len(self.train_loader):

                '''检查该迭代周期是否需要保存或验证'''
                self.checkSaveOrVal(epoch, iteration)

                data = train_iter.next()

                # print(data)

                pretreatmentData = self.pretreatment(data, False)

                modelResult = self.model(*pretreatmentData)

                cost = self.posttreatment(modelResult, pretreatmentData, data)

                self.optimizer.zero_grad()
                # self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()

                loss_avg.add(cost)

                '''
                展示阶段
                在训练的时候仅仅展示在相应阶段的loss
                '''
                if iteration % self.opt.FREQ.SHOW_FREQ == 0:
                    t1 = time.time()
                    print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                          (epoch, self.opt.MODEL.EPOCH, iteration, len(self.train_loader), loss_avg.val(), t1 - t0)),
                    loss_avg.reset()
                    t0 = time.time()

                iteration += 1

            '''动态调整学习率'''
            if self.scheduler != None:
                scheduler.step()
        self.Logger.close_summary()

    def checkSaveOrVal(self, epoch, iteration):
        '''验证'''
        if iteration % self.opt.FREQ.VAL_FREQ == 0:
            self.setModelState('test')
            acc_tmp = self.validate(epoch, iteration)
            '''记录训练结果最大值的模型文件'''
            if acc_tmp > self.highestAcc:
                self.highestAcc = acc_tmp
                torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                    self.opt.ADDRESS.CHECKPOINTS_DIR, str(self.highestAcc)[:6], iteration))

        '''保存'''
        if iteration % self.opt.FREQ.SAVE_FREQ == 0:
            torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                self.opt.ADDRESS.CHECKPOINTS_DIR, self.opt.MODEL.EPOCH, iteration))

        '''恢复训练状态'''
        self.setModelState('train')

    def setModelState(self, state):
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

    def getScheduler(self):
        '''
        需要函数重载，用户指定返回的动态调整lr器
        '''
        pass
        # scheduler_1 = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
        # scheduler_2 = StepLR(self.optimizer, step_size=3, gamma=0.1)

    def finetune(self):
        '''
        微调训练
        一般修改的位置有：
        1.模型参数的 require_grad
        2.optimizer部分的lr
        '''
        return False