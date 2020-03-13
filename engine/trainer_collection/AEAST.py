from engine.trainer import Trainer
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

    def pretreatment(self, data, test=False):
        img, gt = data
        img = img.cuda()
        gt = gt.cuda()
        if test == True:
            return img, gt
        else:
            return (img, )

    def posttreatment(self, modelResult, pretreatmentData, originData, test=False):
        img, gt = originData
        gt = gt.cuda()
        if test == True:
            loss = self.criterion(gt, modelResult)
            return loss
        else:
            loss = self.criterion(gt, modelResult)
            return loss