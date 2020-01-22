from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .roi_align.modules.roi_align import RoIAlign
from torch.utils.data import Dataset, DataLoader
from lib.model.logger import Logger
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from lib.model.networkFactory import networkFactory
import torch.nn.functional as F
import logging
import time
import random
import pickle
from lib.datasets.ctw import CTWDataset
from lib.datasets.syntext import SynthtextDataset
from lib.datasets.totaltext import TotalTextDataset, ToTensor
# from lib.datasets.syntext import SynthtextDataset, ToTensor
from lib.model.focal_loss import FocalLoss
from lib.model.unet.unet_model import UNet
from config.config import config
import cv2


def toNp(x):
    return x.data.cpu().numpy()


def toVar(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

class networkOptimier(object):
    """docstring for TrainModel"""

    def __init__(self, trainDatasetroot, testDatasetroot, modelHome, outputModelHome, outputLogHome,net='vgg16',data='ctw',GPUID = 0,resize_type = 'normal'):
        super(networkOptimier, self).__init__()
        torch.cuda.set_device(GPUID)
        self.strides = [8,16,32,64]
        self.data = data
        self.net = net
        traindataTransform = transforms.Compose([ToTensor(self.strides)])
        if data == 'ctw':
            trainDataset = CTWDataset(trainDatasetroot, traindataTransform,self.strides,istraining=True,resize_type=resize_type)
        elif data == 'synthtext':
            trainDataset = SynthtextDataset(trainDatasetroot, traindataTransform,self.strides,istraining=False,data='synthtext')
        elif data == 'icdar':
            trainDataset = SynthtextDataset(trainDatasetroot, traindataTransform,self.strides,istraining=True,data='icdar')
        elif data == 'totaltext':
            trainDataset = TotalTextDataset(trainDatasetroot, traindataTransform,self.strides,istraining=True,data='totaltext')
        traindataloader = DataLoader(trainDataset, batch_size=1,shuffle=True, num_workers=5)
        # self.dataloader = {'train':traindataloader}

        testdataTransform = transforms.Compose([ToTensor(self.strides)])
        if data == 'ctw':
            testDataset = CTWDataset(testDatasetroot, testdataTransform,self.strides,istraining=False,resize_type=resize_type)
        elif data == 'synthtext':
            testDataset = CTWDataset(testDatasetroot, testdataTransform,self.strides,istraining=False,data='synthtext')
        elif data == 'icdar':
            testDataset = SynthtextDataset(testDatasetroot, testdataTransform,self.strides,istraining=False,data='icdar')
        elif data == 'totaltext':
            testDataset = TotalTextDataset(testDatasetroot, testdataTransform,self.strides,istraining=False,data='totaltext')
        testdataloader = DataLoader(testDataset, batch_size=1,shuffle=False, num_workers=5)
        # self.dataloader = {'test':traindataloader}

        self.dataloader = {'train': traindataloader, 'val': testdataloader}

        # tensorboard log and step lo
        if not os.path.exists(outputLogHome):
            os.makedirs(outputLogHome)
        self.logger = Logger(outputLogHome)
        nf = networkFactory(modelHome)
        if net == 'vgg16':
            self.model = nf.vgg16()
        elif net == 'resnet34':
            self.model = nf.resnet34()
        elif net == 'resnet50':
            self.model = nf.resnet50()
        elif net == 'unet':
            self.model = UNet(3,1)
        elif net == 'resnet50_mask':
            self.model = nf.resnet50_mask()
        print(self.model)
        if not os.path.exists(outputModelHome):
            os.makedirs(outputModelHome)
        if not os.path.exists(outputLogHome):
            os.makedirs(outputLogHome)
        self.outputModelHome = outputModelHome
        self.outputLogHome = outputLogHome
        self.curEpoch = 0
        self.optimizer = None
        # self.focal_loss = FocalLoss()

        self.circle_cls_loss_function = nn.CrossEntropyLoss(ignore_index=-1)
        self.anchor_cls_loss_function = FocalLoss()
        self.mask_loss = FocalLoss()
        self.roi_align = RoIAlign(28, 28, 0)
        self.image_mask_loss = nn.SmoothL1Loss()

    def load(self, modelPath=None):
        if modelPath is not None:
            pretrainedDict = torch.load(modelPath)
            modelDict = self.model.state_dict()
            pretrainedDict = {k: v for k,
                              v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.model.load_state_dict(modelDict)
            print('Load model:{}'.format(modelPath))

    def save(self, modelPath, epoch):
        modelPath = modelPath + '/{}.model'.format(str(epoch))
        print('\nsave model {} in {}'.format(str(epoch), modelPath))
        torch.save(self.model.state_dict(), modelPath)

    def updateLR(self, baselr, epoch, steps, decayRate):
        param = 1
        for step in steps:
            if epoch >= step:
                param += 1
            else:
                break
        for paramGroup in self.optimizer.param_groups:
            paramGroup['lr'] = baselr * decayRate**param
        
        
    def trainval(self, modelPath=None, epoch=0, maxEpoch=1000, baselr=0.001, steps=[1000], decayRate=0.1, valDuration=1000, snapshot=5):
        if modelPath is not None:
            self.load(modelPath)
            self.curEpoch = epoch
        if self.optimizer is None:
            # params = []
            # for param in self.model.parameters():
            #     if param.requires_grad:
            #         params.append(param)
            #     else:
            #         print('No')
            print(self.model.named_parameters())
            self.optimizer = optim.Adam(self.model.parameters(), lr=baselr)
        self.model = self.model.cuda()
        while self.curEpoch < maxEpoch:
            self.curEpoch += 1
            for phase in ['train','val']:
                startEpochTime = time.time()
                Epoch_circle_cls_Loss = {}
                # Epoch_circle_reg_Loss = {}
                # Epoch_anchor_cls_Loss = {}
                # Epoch_anchor_reg_Loss = {}
                Epoch_mask_loss = 0
                Epoch_image_mask_loss = {}
                for stride in self.strides:
                    Epoch_circle_cls_Loss[stride] = 0
                    Epoch_image_mask_loss[stride] = 0
                numOfImage = 0
                startImgTime = time.time()
                datasample = self.dataloader[phase]
                imagenum = datasample.__len__()
                if phase == 'val' and self.curEpoch % valDuration != 0:
                    continue
                for sample in datasample:
                    try:
                        if sample == "FAIL":
                            continue
                        if len(sample) == 0:
                            continue
                        if phase == 'train':
                            isTraining = True
                            self.updateLR(baselr, self.curEpoch,
                                            steps, decayRate)
                            self.model.train(True)
                        else:
                            isTraining = False
                            self.model.eval()
                        numOfImage += 1
                        image = sample['image']

                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        showimage = image.squeeze().numpy().transpose((1, 2, 0))
                        # pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
                        showimage = showimage*std+mean
                        showimage = showimage.astype(np.uint8,copy=True)

                        circle_labels = {}
                        # circle_regres = {}
                        # anchor_labels = {}
                        # anchor_regres = {}
                        # anchor_positive_weights = {}
                        # anchor_negative_weights = {}
                        image = Variable(image.cuda(),requires_grad=False)
                        # mask = sample['mask']
                        # mask = Variable(mask.cuda(),requires_grad=False)
                        # resize_mask = sample['resize_mask']
                        # resize_mask = Variable(resize_mask.cuda(),requires_grad=False)
                        all_circle = {}
                        all_mask = {}
                        mask_gt = {}
                        for stride in self.strides:
                            circle_labels[str(stride)] = Variable(sample['labels_stride_'+str(stride)].squeeze().cuda(),requires_grad=False)
                            all_circle[str(stride)] = Variable(sample['anchors_stride_'+str(stride)].squeeze().cuda(),requires_grad=False)
                            all_mask[str(stride)] = Variable(sample['mask_stride_'+str(stride)].squeeze().cuda(),requires_grad=False)
                            if self.net == 'resnet50_mask':
                                mask_gt[str(stride)] = Variable(sample['mask'+str(stride)].squeeze().cuda(),requires_grad=False)
                        self.optimizer.zero_grad()
                        if self.net == 'resnet50_mask':
                            circle_labels_pred,pred_mask,bbox,bbox_idx,pos_idx_stride,bbox_score,mask_labels= self.model.forward(image,all_circle,circle_labels,threshold=0.4,istraining=isTraining)
                        else:
                            circle_labels_pred,pred_mask,bbox,bbox_idx,pos_idx_stride,bbox_score= self.model.forward(image,all_circle,circle_labels,threshold=0.4,istraining=isTraining) #
                        # backward
                        loss = None

                        losses = {}
                        mask_label = None
                        # losses['image_mask']  = self.image_mask_loss(image_mask,resize_mask)
                        # Epoch_image_mask_loss = Epoch_image_mask_loss+toNp(losses['image_mask'])
                        for stride in self.strides:
                            # circle cls
                            # print(circle_labels_pred[stride],circle_labels[stride])
                            pred_labels = circle_labels_pred[str(stride)]
                            target_labels = circle_labels[str(stride)]
                            label_temploss = None
                            # print(pred_labels,target_labels)
                            if str(stride) in pos_idx_stride:
                                stride_mask = all_mask[str(stride)][pos_idx_stride[str(stride)]]
                                if type(mask_label) == type(None):
                                    mask_label = stride_mask
                                else:
                                    mask_label = torch.cat((mask_label,stride_mask),0)
                            label_temploss = self.anchor_cls_loss_function(pred_labels,target_labels)#self.circle_cls_loss_function(pred_labels,target_labels)
                            # print(label_temploss)
                            if self.net == 'resnet50_mask':
                                losses['seg_'+str(stride)] = F.smooth_l1_loss(mask_labels[str(stride)],mask_gt[str(stride)])
                                Epoch_image_mask_loss[stride] = Epoch_circle_cls_Loss[stride]+toNp(losses['seg_'+str(stride)])
                            losses['cls_'+str(stride)]=label_temploss
                            Epoch_circle_cls_Loss[stride] = Epoch_circle_cls_Loss[stride]+toNp(losses['cls_'+str(stride)])


                        if not type(mask_label) == type(None):
                            mask_label = mask_label.squeeze()
                            ## show mask
                            ## ============
                            pred_mask = pred_mask
                            # print(mask_label.size(),pred_mask.size())
                            losses['mask'] = F.smooth_l1_loss(pred_mask,mask_label)
                            # losses['mask'] = F.cross_entropy(pred_mask,mask_label)
                            Epoch_mask_loss = Epoch_mask_loss+toNp(losses['mask'])
                        for key in losses:
                            if type(loss) == type(None):
                                loss = losses[key]
                            else:
                                loss+=losses[key]

                        # loss = losses['mask']
                            # print(loss)
                        # print(Epoch_circle_cls_Loss,Epoch_circle_reg_Loss)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            # torch.cuda.empty_cache()
                        else:
                            self.optimizer.zero_grad()
                            
                            # del loss,mask_label,circle_labels_pred,pred_mask,bbox,bbox_idx,pos_idx_stride,bbox_score

                        # print(self.curEpoch,losses)
                        print('\rnum:{}/{}'.format(str(numOfImage).zfill(3),imagenum)+"    time:"+str(round(time.time()-startImgTime, 2))+"  "+str(image.size(2))+"*"+str(image.size(3))+"   ",end='')
                        if self.data == 'synthtext':
                            if numOfImage%10000 == 0:
                                self.save(self.outputModelHome, (numOfImage*self.curEpoch)//10000)
                                endEpochTime = time.time()
                                if phase == 'train':
                                    print("\n=================Epoch {} time:{:.2f}===============\n".format(
                                        self.curEpoch, endEpochTime-startEpochTime))
                                else:
                                    print("\n==================Test {:.2f}==================\n".format(
                                        endEpochTime-startEpochTime))
                                startEpochTime = endEpochTime
                                print("time:"+str(round(time.time()-startImgTime, 2))+ \
                                "\ncircle_cls_loss:{}  {}  {}  {}".format(str(Epoch_circle_cls_Loss[8]).zfill(3),str(Epoch_circle_cls_Loss[16]).zfill(3),str(Epoch_circle_cls_Loss[32]).zfill(3),str(Epoch_circle_cls_Loss[64]).zfill(3))+\
                                "\nmask_loss:{}".format(str(Epoch_mask_loss).zfill(3)))
                                self.logger.scalar_summary(phase+'_mask_'+str(stride), Epoch_mask_loss,(numOfImage*self.curEpoch)//10000)
                                for stride in self.strides:
                                    self.logger.scalar_summary(phase+'_circle_cls_'+str(stride), Epoch_circle_cls_Loss[stride], (numOfImage*self.curEpoch)//10000)
                                Epoch_circle_cls_Loss = {}
                                for stride in self.strides:
                                    Epoch_circle_cls_Loss[stride] = 0
                                Epoch_mask_loss=0
                    except:
                        continue
                            
                if self.curEpoch % snapshot == 0 and phase == 'train':
                    self.save(self.outputModelHome, self.curEpoch)
                endEpochTime = time.time()
                if phase == 'train':
                    print("\n=================Epoch {} time:{:.2f}===============\n".format(
                        self.curEpoch, endEpochTime-startEpochTime))
                else:
                    print("\n==================Test {:.2f}==================\n".format(
                        endEpochTime-startEpochTime))
                startEpochTime = endEpochTime

                if self.net == 'resnet50_mask':
                    print("time:"+str(round(time.time()-startImgTime, 2))+ \
                    "\ncircle_cls_loss:{}  {}  {}  {}".format(str(Epoch_circle_cls_Loss[8]).zfill(3),str(Epoch_circle_cls_Loss[16]).zfill(3),str(Epoch_circle_cls_Loss[32]).zfill(3),str(Epoch_circle_cls_Loss[64]).zfill(3))+\
                    "\nmask_strid_loss:{}  {}  {}  {}".format(str(Epoch_image_mask_loss[8]).zfill(3),str(Epoch_image_mask_loss[16]).zfill(3),str(Epoch_image_mask_loss[32]).zfill(3),str(Epoch_image_mask_loss[64]).zfill(3))+\
                    "\nmask_loss:{}".format(str(Epoch_mask_loss).zfill(3)))
                else:           
                    print("time:"+str(round(time.time()-startImgTime, 2))+ \
                    "\ncircle_cls_loss:{}  {}  {}  {}".format(str(Epoch_circle_cls_Loss[8]).zfill(3),str(Epoch_circle_cls_Loss[16]).zfill(3),str(Epoch_circle_cls_Loss[32]).zfill(3),str(Epoch_circle_cls_Loss[64]).zfill(3))+\
                    "\nmask_loss:{}".format(str(Epoch_mask_loss).zfill(3)))
                # "\nimage_mask:{}".format(str(Epoch_image_mask_loss).zfill(3)))
                # "\nanchor_cls_loss:{}  {}  {}  {}  {}".format(str(Epoch_anchor_cls_Loss[8]).zfill(3),str(Epoch_anchor_cls_Loss[16]).zfill(3),str(Epoch_anchor_cls_Loss[32]).zfill(3),str(Epoch_anchor_cls_Loss[64]).zfill(3),str(Epoch_anchor_cls_Loss[128]).zfill(3))+ \
                # "\nanchor_reg_loss:{}  {}  {}  {}  {}".format(str(Epoch_anchor_reg_Loss[8]).zfill(3),str(Epoch_anchor_reg_Loss[16]).zfill(3),str(Epoch_anchor_reg_Loss[32]).zfill(3),str(Epoch_anchor_reg_Loss[64]).zfill(3),str(Epoch_anchor_reg_Loss[128]).zfill(3)))
                # print("\npositive Loss:{}".format(posEpochLoss))
                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                print('save Loss!')
                self.logger.scalar_summary(phase+'_mask_'+str(stride), Epoch_mask_loss, self.curEpoch)
                for stride in self.strides:
                    self.logger.scalar_summary(phase+'_circle_cls_'+str(stride), Epoch_circle_cls_Loss[stride], self.curEpoch)
                torch.cuda.empty_cache()
                    # self.logger.scalar_summary(phase+'_circle_reg_'+str(stride), Epoch_circle_reg_Loss[stride], self.curEpoch)
                    # self.logger.scalar_summary(phase+'_anchor_cls_'+str(stride), Epoch_anchor_cls_Loss[stride], self.curEpoch)
                    # self.logger.scalar_summary(phase+'_anchor_reg_'+str(stride), Epoch_anchor_reg_Loss[stride], self.curEpoch)
                # (2) Log values and gradients of the parameters (histogram)


if __name__ == '__main__':
    trainDatasetroot = config.trainDatasetroot#'/home/shf/fudan_ocr_system/datasets/ICDAR15/Text_Localization/test'
    testDatasetroot = config.testDatasetroot#'/home/shf/fudan_ocr_system/datasets/ICDAR15/Text_Localization/train'
    modelHome = config.modelHome#'/home/shf/fudan_ocr_system/LSN/pretrainmodel'
    outputModelHome = config.outputModelHome#'/home/shf/fudan_ocr_system/LSN/lib/model/data/2019AAAI/output/resnet50/outputmodel'
    outputLogHome = config.outputLogHome#'/home/shf/fudan_ocr_system/LSN/lib/model/data/2019AAAI/output/resnet50/outputlog'
    no = networkOptimier(trainDatasetroot, testDatasetroot, modelHome, outputModelHome, outputLogHome)
    no.trainval()
