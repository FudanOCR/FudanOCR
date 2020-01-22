import time
from utils import utils
import torch
from torch.autograd import Variable

class Trainer(object):

    def __init__(self,model,train_loader,opt,criterion,optimizer,alphabet,nc):

        self.opt = opt
        self.criterion = criterion
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_avg = utils.averager()
        self.converter = utils.strLabelConverterForAttention(alphabet)
        self.nc = nc

    def trainBatch(self,train_iter):

        image = torch.FloatTensor(self.opt.batchSize, self.nc, self.opt.imgH, self.opt.imgW)
        text = torch.LongTensor(self.opt.batchSize * 5)
        text_rev = torch.LongTensor(self.opt.batchSize * 5)
        length = torch.IntTensor(self.opt.batchSize)

        if self.opt.cuda:
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=range(self.opt.ngpu))
            image = image.cuda()
            text = text.cuda()
            text_rev = text_rev.cuda()
            self.criterion = self.criterion.cuda()

        image = Variable(image)
        text = Variable(text)
        text_rev = Variable(text_rev)
        length = Variable(length)




        data = train_iter.next()
        if self.opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            t_rev, _ = self.converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = self.model(image, length, text, text_rev)
            cost = self.criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = self.converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = self.model(image, length, text, text_rev)
            cost = self.criterion(preds, text)

        self.model.zero_grad()
        cost.backward()
        self.optimizer.step()
        return cost

    def train(self):
        t0 = time.time()
        acc = 0
        acc_tmp = 0
        for epoch in range(self.opt.niter):

            train_iter = iter(self.train_loader)
            i = 0
            while i < len(self.train_loader):
                # print("main函数里,可迭代次数为 %d" %  len(train_loader))

                # if i % self.opt.valInterval == 0:
                #     for p in self.model.parameters():
                #         p.requires_grad = False
                #     self.model.eval()
                #
                #     acc_tmp = val(test_dataset, criterion)
                #     if acc_tmp > acc:
                #         acc = acc_tmp
                #         torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                #             self.opt.experiment, i, str(acc)[:6]))

                if i % self.opt.saveInterval == 0:
                    torch.save(self.model.state_dict(), '{0}/{1}_{2}.pth'.format(
                        self.opt.experiment, epoch, i))

                for p in self.model.parameters():
                    p.requires_grad = True
                self.model.train()

                cost = self.trainBatch(train_iter)
                self.loss_avg.add(cost)

                if i % self.opt.displayInterval == 0:
                    t1 = time.time()
                    print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                          (epoch, self.opt.niter, i, len(self.train_loader), self.loss_avg.val(), t1 - t0)),
                    self.loss_avg.reset()
                    t0 = time.time()

                i += 1