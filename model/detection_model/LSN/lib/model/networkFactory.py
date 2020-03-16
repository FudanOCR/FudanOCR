from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .roi_align.modules.roi_align import RoIAlign
import torch
from torchvision import models
import os
import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s')

class VGG16(nn.Module):
    def __init__(self,modelRoot):
        super(VGG16,self).__init__()
        #Init Model Structure
        vgg16 = models.vgg16(pretrained = False)
        # print(vgg16)
        vgg16.load_state_dict(torch.load(modelRoot+'/'+'vgg16.pth'))
        self.RCNN_base = vgg16.features
        self.RCNN_expend = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1), ceil_mode=False),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        self.stride8_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride16_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride32_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride64_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride8_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))
        self.stride16_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))
        self.stride32_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))

        self.RCNN_deconv = nn.Sequential(
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)) # concate
        self.mask_generate = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.roi_align = RoIAlign(14, 14, 0)
        self._init_weights()


    def forward(self,x,all_circle,gt_labels,istraining = True,threshold=0.5,testMaxNum=100):
        concate_conv = {}
        circle_cls = {8:self.stride8_circle_cls,16:self.stride16_circle_cls,32:self.stride32_circle_cls,64:self.stride64_circle_cls}
        #stride_conv = {8:self.stride8_conv,16:self.stride16_conv,32:self.stride32_conv,64:self.stride64_conv}
        for idx,module in enumerate(self.RCNN_base._modules.values()):
            x = module(x)
            # print(idx)
            if idx == 22:
                concate_conv[8] = x
            if idx == 29:
                concate_conv[16] = x
        # print(self.RCNN_expend)
        for idx,modulex in enumerate(self.RCNN_expend._modules.values()):
            x = modulex(x)
            if idx == 3:
                concate_conv[32] = x
        conv = {}
        conv[64] = x
        for idx,modulex in enumerate(self.RCNN_deconv._modules.values()):
            x = modulex(x)
            if idx == 5:
                x = torch.cat((x,self.stride32_concate(concate_conv[32])),1)
                conv[32] = x
            if idx == 11:
                x = torch.cat((x,self.stride16_concate(concate_conv[16])),1)
                conv[16] = x
            if idx == 17:
                x = torch.cat((x,self.stride8_concate(concate_conv[8])),1)
                conv[8] = x
        circle_labels = {}
        # del concate_conv
        strides = [8,16,32,64]
        mask_conv = None
        # image_mask = self.image_mask_generate(mask_conv)
        # print(mask_conv.size())
        # circle_regres = {}
        # anchor_labels = {}
        # anchor_regres = {}
        
        mask_feature = None
        mask_conv_feature = None
        roi = None
        bbox_idx = None
        pos_idx_stride = {}
        bbox_score = None
        for stride in strides:
            # circle classify
            stride_circle_cls = None
            stride_circle_cls = circle_cls[stride](conv[stride])
            n,c,h,w = stride_circle_cls.size()
            stride_circle_cls = stride_circle_cls.view(n,2,int(c/2),h,w)
            circle_labels[str(stride)] = stride_circle_cls.permute(0,3,4,2,1).contiguous().view(n*int(c/2)*h*w,2)

            # pred_labels = circle_labels[str(stride)]
            # prod = nn.functional.softmax(pred_labels)
            ## roi
            if istraining:
                sort_score,sort_idx = torch.sort(gt_labels[str(stride)],descending=True)
                postive_num = int(torch.sum(gt_labels[str(stride)])+1)
                # print(postive_num)
                select_postive_num = max(1,min(80,postive_num))
                select_negtive_num = max(1,int(select_postive_num/4))
                postive_idx = np.random.randint(0,postive_num,size=select_postive_num)
                negtive_idx = np.random.randint(postive_num,gt_labels[str(stride)].size(0),size=select_negtive_num)
                select_idx = torch.from_numpy(np.concatenate((postive_idx,negtive_idx))).cuda()
                pos_idx = sort_idx[select_idx]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[select_idx]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[select_idx]),0)
            else:
                pred_labels = circle_labels[str(stride)]
                prod = nn.functional.softmax(pred_labels)
                numpy_prod = prod.data.cpu().numpy()
                length = min(len(np.where(numpy_prod[:,1]>=threshold)[0]),testMaxNum)
                print(length)
                score = prod[:,1]
                sort_score,sort_idx = torch.sort(score,descending=True)
                # num = int(torch.sum(sort_score>=threshold).data.cpu())+1
                num = length+1
                # print(stride,num,length,sort_score[length-1].data.cpu())
                pos_idx = sort_idx[:num]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[:num]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[:num]),0)

            # image = np.zeros((1024,1024,3),dtype=np.uint8)
            # # print(image)
            # image[:,:,:] = 255
            # for box in bbox:
            #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     cv2.imshow('image',image)
            #     cv2.waitKey(0)
            # print(roi_bbox)
            if type(roi) == type(None):
                roi = bbox
            else:
                roi = torch.cat((roi,bbox),0)
        

        bbox_idx = torch.IntTensor(len(roi))
        bbox_idx = Variable(bbox_idx.fill_(0).cuda(),requires_grad=False)
        roi_bbox = roi*1.0/8
        mask_feature = self.roi_align(conv[8],roi_bbox,bbox_idx)

        if type(mask_feature) == type(None):
            pred_mask = None
        else:
            # print(mask_feature)
            pred_mask = self.mask_generate(mask_feature).squeeze()#.permute(0,2,3,1).contiguous()
            # print(pred_mask)
        # del conv
        return circle_labels,pred_mask,roi,bbox_idx,pos_idx_stride,bbox_score#,circle_regres#,anchor_labels,anchor_regres
    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            for name,param in m.named_parameters():
                param.requires_grad = True
                if name.split('.')[-1] == "weight":
                    nn.init.normal(m.state_dict()[name], mean, stddev)
                elif name.split('.')[-1] == "bias":
                    nn.init.constant(m.state_dict()[name], mean)

        normal_init(self.RCNN_expend, 0, 0.01)
        normal_init(self.stride8_circle_cls, 0, 0.01)
        normal_init(self.stride16_circle_cls, 0, 0.01)
        normal_init(self.stride32_circle_cls, 0, 0.01)
        normal_init(self.stride64_circle_cls, 0, 0.01)

        normal_init(self.stride8_concate, 0, 0.01)
        normal_init(self.stride16_concate, 0, 0.01)
        normal_init(self.stride32_concate, 0, 0.01)
        normal_init(self.mask_generate, 0, 0.01)
        
        normal_init(self.RCNN_deconv, 0, 0.01)
        print('init success!')
        
class ResNet34(nn.Module):
    def __init__(self,modelRoot):
        super(ResNet34,self).__init__()
        #Init Model Structure
        # print(vgg16)
        print('resnet')
        resnet34 = models.resnet34(pretrained = False)
        resnet34.load_state_dict(torch.load(modelRoot+'/'+'resnet34.pth'))
        modules = list(resnet34.children())[:-2]
        
        self.RCNN_base  = nn.Sequential(*modules)
        # print(self.RCNN_base)
        self.RCNN_expend = nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1), ceil_mode=False),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))

        self.stride8_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride16_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride32_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride64_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride8_concate = nn.Sequential(nn.Conv2d(128,256,kernel_size=1,padding=0))
        self.stride16_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))
        self.stride32_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))

        self.RCNN_deconv = nn.Sequential(
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)) # concate
        self.mask_generate = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.roi_align = RoIAlign(14, 14, 0)
        self._init_weights()


    def forward(self,x,all_circle,gt_labels,istraining = True,threshold=0.5,testMaxNum=100):
        concate_conv = {}
        # print(x.size())
        circle_cls = {8:self.stride8_circle_cls,16:self.stride16_circle_cls,32:self.stride32_circle_cls,64:self.stride64_circle_cls}
        #stride_conv = {8:self.stride8_conv,16:self.stride16_conv,32:self.stride32_conv,64:self.stride64_conv}
        for idx,module in enumerate(self.RCNN_base._modules.values()):
            x = module(x)
            if idx == 5:
                concate_conv[8] = x
            if idx == 6:
                concate_conv[16] = x
            if idx == 7:
                concate_conv[32] = x
        x = self.RCNN_expend(x)    
        conv = {}
        # print(x.size())
        conv[64] = x
        for idx,modulex in enumerate(self.RCNN_deconv._modules.values()):
            x = modulex(x)
            if idx == 5:
                x = torch.cat((x,self.stride32_concate(concate_conv[32])),1)
                conv[32] = x
            if idx == 11:
                x = torch.cat((x,concate_conv[16]),1)
                conv[16] = x
            if idx == 17:
                x = torch.cat((x,self.stride8_concate(concate_conv[8])),1)
                conv[8] = x
        circle_labels = {}
        # del concate_conv
        strides = [8,16,32,64]
        mask_conv = None
        # image_mask = self.image_mask_generate(mask_conv)
        # print(mask_conv.size())
        # circle_regres = {}
        # anchor_labels = {}
        # anchor_regres = {}
        
        mask_feature = None
        mask_conv_feature = None
        roi = None
        bbox_idx = None
        pos_idx_stride = {}
        bbox_score = None
        for stride in strides:
            # circle classify
            stride_circle_cls = None
            stride_circle_cls = circle_cls[stride](conv[stride])
            n,c,h,w = stride_circle_cls.size()
            stride_circle_cls = stride_circle_cls.view(n,2,int(c/2),h,w)
            circle_labels[str(stride)] = stride_circle_cls.permute(0,3,4,2,1).contiguous().view(n*int(c/2)*h*w,2)

            
            ## roi
            if istraining:
                sort_score,sort_idx = torch.sort(gt_labels[str(stride)],descending=True)
                postive_num = int(torch.sum(gt_labels[str(stride)])+1)
                # print(postive_num)
                select_postive_num = max(1,min(100,postive_num))
                # select_negtive_num = max(1,int(select_postive_num/4))
                postive_idx = np.random.randint(0,postive_num,size=select_postive_num)
                # negtive_idx = np.random.randint(postive_num,gt_labels[str(stride)].size(0),size=select_negtive_num)
                select_idx = torch.from_numpy(postive_idx).cuda()
                pos_idx = sort_idx[select_idx]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[select_idx]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[select_idx]),0)
            else:
                pred_labels = circle_labels[str(stride)]
                prod = nn.functional.softmax(pred_labels)
                numpy_prod = prod.data.cpu().numpy()
                length = min(len(np.where(numpy_prod[:,1]>=threshold)[0]),testMaxNum)
                print(length)
                score = prod[:,1]
                sort_score,sort_idx = torch.sort(score,descending=True)
                # num = int(torch.sum(sort_score>=threshold).data.cpu())+1
                num = length+1
                # print(stride,num,length,sort_score[length-1].data.cpu())
                pos_idx = sort_idx[:num]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[:num]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[:num]),0)

            # image = np.zeros((1024,1024,3),dtype=np.uint8)
            # # print(image)
            # image[:,:,:] = 255
            # for box in bbox:
            #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     cv2.imshow('image',image)
            #     cv2.waitKey(0)
            # print(roi_bbox)
            if type(roi) == type(None):
                roi = bbox
            else:
                roi = torch.cat((roi,bbox),0)

        bbox_idx = torch.IntTensor(len(roi))
        bbox_idx = Variable(bbox_idx.fill_(0).cuda(),requires_grad=False)
        roi_bbox = roi*1.0/8
        mask_feature = self.roi_align(conv[8],roi_bbox,bbox_idx)

        if type(mask_feature) == type(None):
            pred_mask = None
        else:
            # print(mask_feature)
            pred_mask = self.mask_generate(mask_feature).squeeze()#.permute(0,2,3,1).contiguous()
            # print(pred_mask)
        # del conv
        return circle_labels,pred_mask,roi,bbox_idx,pos_idx_stride,bbox_score#,circle_regres#,anchor_labels,anchor_regres
    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            for name,param in m.named_parameters():
                param.requires_grad = True
                if name.split('.')[-1] == "weight":
                    nn.init.normal(m.state_dict()[name], mean, stddev)
                elif name.split('.')[-1] == "bias":
                    nn.init.constant(m.state_dict()[name], mean)

        normal_init(self.RCNN_expend, 0, 0.01)
        normal_init(self.stride8_circle_cls, 0, 0.01)
        normal_init(self.stride16_circle_cls, 0, 0.01)
        normal_init(self.stride32_circle_cls, 0, 0.01)
        normal_init(self.stride64_circle_cls, 0, 0.01)

        normal_init(self.stride8_concate, 0, 0.01)
        normal_init(self.stride16_concate, 0, 0.01)
        normal_init(self.stride32_concate, 0, 0.01)
        normal_init(self.mask_generate, 0, 0.01)
        
        normal_init(self.RCNN_deconv, 0, 0.01)
        print('init success!')


class ResNet50(nn.Module):
    def __init__(self,cfg):
        super(ResNet50,self).__init__()
        #Init Model Structure
        # print(vgg16)
        print('resnet50')
        resnet = models.resnet50(pretrained = False)
        # resnet.load_state_dict(torch.load(cfg.LSN.modelPath))
        modules = list(resnet.children())[:-2]
        
        self.RCNN_base  = nn.Sequential(*modules)
        # print(self.RCNN_base)
        self.RCNN_expend = nn.Sequential(nn.Conv2d(2048,512,kernel_size=1,padding=0),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1), ceil_mode=False),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))

        self.stride8_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride16_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride32_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride64_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))

        self.stride8_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))
        self.stride16_concate = nn.Sequential(nn.Conv2d(1024,256,kernel_size=1,padding=0))
        self.stride32_concate = nn.Sequential(nn.Conv2d(2048,256,kernel_size=1,padding=0))
        self.RCNN_deconv = nn.Sequential(
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)) # concate
        self.mask_generate = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.roi_align = RoIAlign(14, 14, 0)
        self._init_weights()


    def forward(self,x, all_circle,  gt_labels, istraining = True,threshold=0.5,testMaxNum=100):
        concate_conv = {}
        # print(x.size())
        circle_cls = {8:self.stride8_circle_cls,16:self.stride16_circle_cls,32:self.stride32_circle_cls,64:self.stride64_circle_cls}
        #stride_conv = {8:self.stride8_conv,16:self.stride16_conv,32:self.stride32_conv,64:self.stride64_conv}
        for idx,module in enumerate(self.RCNN_base._modules.values()):
            x = module(x)
            if idx == 5:
                concate_conv[8] = x
            if idx == 6:
                concate_conv[16] = x
            if idx == 7:
                concate_conv[32] = x
        x = self.RCNN_expend(x) 
        # print(concate_conv[16].size())
        conv = {}
        # print(x.size())
        conv[64] = x
        for idx,modulex in enumerate(self.RCNN_deconv._modules.values()):
            # print(idx,x.size())
            x = modulex(x)
            if idx == 5:
                x = torch.cat((x,self.stride32_concate(concate_conv[32])),1)
                conv[32] = x
            if idx == 11:
                x = torch.cat((x,self.stride16_concate(concate_conv[16])),1)
                conv[16] = x
            if idx == 17:
                x = torch.cat((x,self.stride8_concate(concate_conv[8])),1)
                conv[8] = x
        circle_labels = {}
        # del concate_conv
        strides = [8,16,32,64]
        mask_conv = None
        # image_mask = self.image_mask_generate(mask_conv)
        # print(mask_conv.size())
        # circle_regres = {}
        # anchor_labels = {}
        # anchor_regres = {}
        
        mask_feature = None
        mask_conv_feature = None
        roi = None
        bbox_idx = None
        pos_idx_stride = {}
        bbox_score = None
        _,_,H,W = conv[8].size()
        roi_feature = F.upsample(conv[64],size=(H,W),mode='bilinear')+F.upsample(conv[32],size=(H,W),mode='bilinear')+F.upsample(conv[16],size=(H,W),mode='bilinear')+conv[8]
        for stride in strides:
            # circle classify
            stride_circle_cls = None
            # print("circle_cls:",circle_cls)
            # print("conv:", conv)
            stride_circle_cls = circle_cls[stride](conv[stride])
            n,c,h,w = stride_circle_cls.size()
            stride_circle_cls = stride_circle_cls.view(n,2,int(c/2),h,w)
            circle_labels[str(stride)] = stride_circle_cls.permute(0,3,4,2,1).contiguous().view(n*int(c/2)*h*w,2)

            
            ## roi
            if istraining:
                sort_score,sort_idx = torch.sort(gt_labels[str(stride)],descending=True)
                postive_num = int(torch.sum(gt_labels[str(stride)])+1)
                # print(postive_num)
                select_postive_num = max(1,min(50,postive_num))
                # select_negtive_num = max(1,int(select_postive_num/4))
                postive_idx = np.random.randint(0,postive_num,size=select_postive_num)
                # negtive_idx = np.random.randint(postive_num,gt_labels[str(stride)].size(0),size=select_negtive_num)
                select_idx = torch.from_numpy(postive_idx).cuda()
                pos_idx = sort_idx[select_idx]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[select_idx]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[select_idx]),0)
            else:
                pred_labels = circle_labels[str(stride)]
                prod = nn.functional.softmax(pred_labels)
                numpy_prod = prod.data.cpu().numpy()
                length = min(len(np.where(numpy_prod[:,1]>=threshold)[0]),testMaxNum)
                score = prod[:,1]
                sort_score,sort_idx = torch.sort(score,descending=True)
                # num = int(torch.sum(sort_score>=threshold).data.cpu())+1
                num = length+1
                print(num)
                # print(stride,num,length,sort_score[length-1].data.cpu())
                pos_idx = sort_idx[:num]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[:num]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[:num]),0)

            # image = np.zeros((1024,1024,3),dtype=np.uint8)
            # # print(image)
            # image[:,:,:] = 255
            # for box in bbox:
            #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     cv2.imshow('image',image)
            #     cv2.waitKey(0)
            # print(roi_bbox)
            if type(roi) == type(None):
                roi = bbox
            else:
                roi = torch.cat((roi,bbox),0)

        bbox_idx = torch.IntTensor(len(roi))
        bbox_idx = Variable(bbox_idx.fill_(0).cuda(),requires_grad=False)
        roi_bbox = roi*1.0/8
        mask_feature = self.roi_align(roi_feature,roi_bbox,bbox_idx)

        if type(mask_feature) == type(None):
            pred_mask = None
        else:
            # print(mask_feature)
            pred_mask = self.mask_generate(mask_feature).squeeze()#.permute(0,2,3,1).contiguous()
            # print(pred_mask)
        # del conv
        return circle_labels,pred_mask,roi,bbox_idx,pos_idx_stride,bbox_score#,circle_regres#,anchor_labels,anchor_regres
    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            for name,param in m.named_parameters():
                param.requires_grad = True
                if name.split('.')[-1] == "weight":
                    nn.init.normal(m.state_dict()[name], mean, stddev)
                elif name.split('.')[-1] == "bias":
                    nn.init.constant(m.state_dict()[name], mean)

        normal_init(self.RCNN_expend, 0, 0.01)
        normal_init(self.stride8_circle_cls, 0, 0.01)
        normal_init(self.stride16_circle_cls, 0, 0.01)
        normal_init(self.stride32_circle_cls, 0, 0.01)
        normal_init(self.stride64_circle_cls, 0, 0.01)

        normal_init(self.stride8_concate, 0, 0.01)
        normal_init(self.stride16_concate, 0, 0.01)
        normal_init(self.stride32_concate, 0, 0.01)
        normal_init(self.mask_generate, 0, 0.01)
        
        normal_init(self.RCNN_deconv, 0, 0.01)
        print('init success!')

class ResNet50_mask(nn.Module):
    def __init__(self,modelRoot):
        super(ResNet50_mask,self).__init__()
        #Init Model Structure
        # print(vgg16)
        print('resnet50')
        resnet = models.resnet50(pretrained = False)
        resnet.load_state_dict(torch.load(modelRoot+'/'+'resnet50.pth'))
        modules = list(resnet.children())[:-2]
        
        self.RCNN_base  = nn.Sequential(*modules)
        # print(self.RCNN_base)
        self.RCNN_expend = nn.Sequential(nn.Conv2d(2048,512,kernel_size=1,padding=0),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.MaxPool2d(kernel_size=2, stride=2, dilation=(1, 1), ceil_mode=False),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))

        self.stride8_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride16_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride32_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride64_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))

        self.stride8_mask = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,1,kernel_size=1,padding=0))
        self.stride16_mask = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,1,kernel_size=1,padding=0))
        self.stride32_mask = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,1,kernel_size=1,padding=0))
        self.stride64_mask = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,1,kernel_size=1,padding=0))

        self.stride8_concate = nn.Sequential(nn.Conv2d(512,256,kernel_size=1,padding=0))
        self.stride16_concate = nn.Sequential(nn.Conv2d(1024,256,kernel_size=1,padding=0))
        self.stride32_concate = nn.Sequential(nn.Conv2d(2048,256,kernel_size=1,padding=0))
        self.RCNN_deconv = nn.Sequential(
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # concate
            nn.ConvTranspose2d(512,512,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)) # concate
        self.mask_generate = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,kernel_size=1,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.roi_align = RoIAlign(14, 14, 0)
        self._init_weights()


    def forward(self,x,all_circle,gt_labels,istraining = True,threshold=0.5,testMaxNum=100):
        concate_conv = {}
        # print(x.size())
        circle_cls = {8:self.stride8_circle_cls,16:self.stride16_circle_cls,32:self.stride32_circle_cls,64:self.stride64_circle_cls}
        mask_cls = {8:self.stride8_mask,16:self.stride16_mask,32:self.stride32_mask,64:self.stride64_mask}
        #stride_conv = {8:self.stride8_conv,16:self.stride16_conv,32:self.stride32_conv,64:self.stride64_conv}
        for idx,module in enumerate(self.RCNN_base._modules.values()):
            x = module(x)
            if idx == 5:
                concate_conv[8] = x
            if idx == 6:
                concate_conv[16] = x
            if idx == 7:
                concate_conv[32] = x
        x = self.RCNN_expend(x) 
        # print(concate_conv[16].size())
        conv = {}
        # print(x.size())
        conv[64] = x
        for idx,modulex in enumerate(self.RCNN_deconv._modules.values()):
            # print(idx,x.size())
            x = modulex(x)
            if idx == 5:
                x = torch.cat((x,self.stride32_concate(concate_conv[32])),1)
                conv[32] = x
            if idx == 11:
                x = torch.cat((x,self.stride16_concate(concate_conv[16])),1)
                conv[16] = x
            if idx == 17:
                x = torch.cat((x,self.stride8_concate(concate_conv[8])),1)
                conv[8] = x
        circle_labels = {}
        mask_labels = {}
        # del concate_conv
        strides = [8,16,32,64]
        mask_conv = None
        # image_mask = self.image_mask_generate(mask_conv)
        # print(mask_conv.size())
        # circle_regres = {}
        # anchor_labels = {}
        # anchor_regres = {}
        
        mask_feature = None
        mask_conv_feature = None
        roi = None
        bbox_idx = None
        pos_idx_stride = {}
        bbox_score = None
        _,_,H,W = conv[8].size()
        roi_feature = F.upsample(conv[64],size=(H,W),mode='bilinear')+F.upsample(conv[32],size=(H,W),mode='bilinear')+F.upsample(conv[16],size=(H,W),mode='bilinear')+conv[8]
        for stride in strides:
            # circle classify
            stride_mask_cls = mask_cls[stride](conv[stride])
            stride_circle_cls = None
            stride_circle_cls = circle_cls[stride](conv[stride]+stride_mask_cls)
            n,c,h,w = stride_circle_cls.size()
            stride_circle_cls = stride_circle_cls.view(n,2,int(c/2),h,w)
            circle_labels[str(stride)] = stride_circle_cls.permute(0,3,4,2,1).contiguous().view(n*int(c/2)*h*w,2)
            mask_labels[str(stride)] = stride_mask_cls.squeeze()
            ## roi
            if istraining:
                sort_score,sort_idx = torch.sort(gt_labels[str(stride)],descending=True)
                postive_num = int(torch.sum(gt_labels[str(stride)])+1)
                # print(postive_num)
                select_postive_num = max(1,min(100,postive_num))
                # select_negtive_num = max(1,int(select_postive_num/4))
                postive_idx = np.random.randint(0,postive_num,size=select_postive_num)
                # negtive_idx = np.random.randint(postive_num,gt_labels[str(stride)].size(0),size=select_negtive_num)
                select_idx = torch.from_numpy(postive_idx).cuda()
                pos_idx = sort_idx[select_idx]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[select_idx]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[select_idx]),0)
            else:
                pred_labels = circle_labels[str(stride)]
                prod = nn.functional.softmax(pred_labels)
                numpy_prod = prod.data.cpu().numpy()
                length = min(len(np.where(numpy_prod[:,1]>=threshold)[0]),testMaxNum)
                score = prod[:,1]
                sort_score,sort_idx = torch.sort(score,descending=True)
                # num = int(torch.sum(sort_score>=threshold).data.cpu())+1
                num = length+1
                # print(stride,num,length,sort_score[length-1].data.cpu())
                pos_idx = sort_idx[:num]
                bbox = all_circle[str(stride)][pos_idx,:]
                pos_idx_stride[str(stride)] = pos_idx
                if type(bbox_score) == type(None):
                    bbox_score = sort_score[:num]
                else:
                    bbox_score = torch.cat((bbox_score,sort_score[:num]),0)

            # image = np.zeros((1024,1024,3),dtype=np.uint8)
            # # print(image)
            # image[:,:,:] = 255
            # for box in bbox:
            #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     cv2.imshow('image',image)
            #     cv2.waitKey(0)
            # print(roi_bbox)
            if type(roi) == type(None):
                roi = bbox
            else:
                roi = torch.cat((roi,bbox),0)

        bbox_idx = torch.IntTensor(len(roi))
        bbox_idx = Variable(bbox_idx.fill_(0).cuda(),requires_grad=False)
        roi_bbox = roi*1.0/8
        mask_feature = self.roi_align(roi_feature,roi_bbox,bbox_idx)

        if type(mask_feature) == type(None):
            pred_mask = None
        else:
            # print(mask_feature)
            pred_mask = self.mask_generate(mask_feature).squeeze()#.permute(0,2,3,1).contiguous()
            # print(pred_mask)
        # del conv
        return circle_labels,pred_mask,roi,bbox_idx,pos_idx_stride,bbox_score,mask_labels#,circle_regres#,anchor_labels,anchor_regres
    
    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            for name,param in m.named_parameters():
                param.requires_grad = True
                if name.split('.')[-1] == "weight":
                    nn.init.normal(m.state_dict()[name], mean, stddev)
                elif name.split('.')[-1] == "bias":
                    nn.init.constant(m.state_dict()[name], mean)

        normal_init(self.RCNN_expend, 0, 0.01)
        normal_init(self.stride8_circle_cls, 0, 0.01)
        normal_init(self.stride16_circle_cls, 0, 0.01)
        normal_init(self.stride32_circle_cls, 0, 0.01)
        normal_init(self.stride64_circle_cls, 0, 0.01)

        normal_init(self.stride8_mask, 0, 0.01)
        normal_init(self.stride16_mask, 0, 0.01)
        normal_init(self.stride32_mask, 0, 0.01)
        normal_init(self.stride64_mask, 0, 0.01)

        normal_init(self.stride8_concate, 0, 0.01)
        normal_init(self.stride16_concate, 0, 0.01)
        normal_init(self.stride32_concate, 0, 0.01)
        normal_init(self.mask_generate, 0, 0.01)
        
        normal_init(self.RCNN_deconv, 0, 0.01)
        print('init success!')

class networkFactory(object):
    """docstring for network_factory"""
    def __init__(self,root):
        super(networkFactory, self).__init__()
        self.modelRoot = root

    def vgg16(self):
        return VGG16(self.modelRoot)

    def resnet34(self):
    	return ResNet34(self.modelRoot)
    
    def resnet50(self):
        return ResNet50(self.modelRoot)
    
    def resnet50_mask(self):
        return ResNet50_mask(self.modelRoot)

if __name__ == '__main__':
    nf = networkFactory(config.modelPath.rstrip('.pth').rstrip(config.net))#('/home/shf/fudan_ocr_system/LSN/pretrainmodel')
    resnet = nf.resnet50().cuda()
    x = resnet.forward(Variable(torch.FloatTensor(1,3,1024,1024)).cuda(),None,None)
    print(x)
    # reg,border,reverse,positive = resnet.forward()
    # print(reg.size(),border.size(),reverse.size(),positive.size())
