from engine.trainer import Trainer
class RARE_Trainer(Trainer):
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
        from utils.loadData import loadData

        image = torch.FloatTensor(self.opt.MODEL.BATCH_SIZE, self.opt.IMAGE.IMG_CHANNEL, self.opt.IMAGE.IMG_H,
                                  self.opt.IMAGE.IMG_H)
        text = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        text_rev = torch.LongTensor(self.opt.MODEL.BATCH_SIZE * 5)
        length = torch.IntTensor(self.opt.MODEL.BATCH_SIZE)

        if self.opt.BASE.CUDA:
            # self.model = torch.nn.DataParallel(self.model, device_ids=range(self.opt.ngpu))
            image = image.cuda()
            text = text.cuda()
            text_rev = text_rev.cuda()
            self.criterion = self.criterion.cuda()

        image = Variable(image)
        text = Variable(text)
        text_rev = Variable(text_rev)
        length = Variable(length)

        cpu_images, cpu_texts = data
        loadData(image, cpu_images)
        t, l = self.converter.encode(cpu_texts, scanned=True)
        loadData(text, t)
        loadData(length, l)
        return image, length, text, text_rev, test
        # preds = self.model(image, length, text, text_rev)
        # cost = self.criterion(preds, text)

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        '''
        将神经网络传出的数据解码为可用于计算结果的数据
        '''
        import torch
        from torch.autograd import Variable

        if test == False:
            image, length, text, text_rev, test = pretreatmentData
            # preds = self.model(image, length, text, text_rev)
            preds = modelResult
            cost = self.criterion(preds, text)
            return cost
        else:
            cpu_images, cpu_texts = originData
            preds = modelResult
            image, length, text, text_rev, _ = pretreatmentData
            cost = self.criterion(preds, text)
            _, preds = preds.max(1)
            preds = preds.view(-1)
            sim_preds = self.converter.decode(preds.data, length.data)

            sim_preds = [i.split('$')[0] + '$'  for i in sim_preds]

            return cost, sim_preds, cpu_texts