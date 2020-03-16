#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging

# python 3 confusing imports :(
from model.detection_model.LSN.lib.model.unet.unet_parts import *
from model.detection_model.LSN.lib.model.roi_align.modules.roi_align import RoIAlign

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.up1 = up(1024, 512)
        self.up2 = up(1024, 512)
        self.up3 = up(1024, 256)
        self.up4 = up(512, 256)
        self.up5 = up(512, 256)
        self.up6 = up(512, 256)
        # self.outc = outconv(64, n_classes)
        self.outc = outconv(64, n_classes)
        # self.outc = outconv(64, n_classes)
        self.stride8_circle_cls = nn.Sequential(nn.Conv2d(256,128,kernel_size=3,padding=1),nn.Conv2d(128,8,kernel_size=1,padding=0))
        self.stride16_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride32_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))
        self.stride64_circle_cls = nn.Sequential(nn.Conv2d(512,256,kernel_size=3,padding=1),nn.Conv2d(256,8,kernel_size=1,padding=0))

        self.mask_generate = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=1,padding=0),
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
        self.roi_align = RoIAlign(14, 14,extrapolation_value=0)
        self._init_weights()

    def forward(self, x,all_circle,gt_labels,istraining = True,threshold=0.5):
        conv = {}
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x4 = self.down3(x)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        conv[64] = self.down6(x6)
        conv[32] = self.up1(conv[64],x6)
        conv[16] = self.up2(conv[32], x5)
        conv[8] = self.up3(conv[16], x4)
        circle_labels = {}
        # del concate_conv
        strides = [8,16,32,64]
        mask_conv = None
        # image_mask = self.image_mask_generate(mask_conv)
        # print(mask_conv.size())
        # circle_regres = {}
        # anchor_labels = {}
        # anchor_regres = {}
        circle_cls = {8:self.stride8_circle_cls,16:self.stride16_circle_cls,32:self.stride32_circle_cls,64:self.stride64_circle_cls}
        mask_feature = None
        mask_conv_feature = None
        roi = None
        bbox_idx = None
        pos_idx_stride = {}
        bbox_score = None
        # _,_,H,W = conv[8].size()
        #roi_feature = F.upsample(conv[64],size=(H,W),mode='bilinear')+F.upsample(conv[32],size=(H,W),mode='bilinear')+F.upsample(conv[16],size=(H,W),mode='bilinear')+conv[8]
        for stride in strides:
            # circle classify
            stride_circle_cls = None
            # print(stride,conv[stride].size())
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
                length = min(len(np.where(numpy_prod[:,1]>=threshold)[0]),1000)
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

        normal_init(self.stride8_circle_cls, 0, 0.01)
        normal_init(self.stride16_circle_cls, 0, 0.01)
        normal_init(self.stride32_circle_cls, 0, 0.01)
        normal_init(self.stride64_circle_cls, 0, 0.01)

        # normal_init(self.stride8_concate, 0, 0.01)
        # normal_init(self.stride16_concate, 0, 0.01)
        # normal_init(self.stride32_concate, 0, 0.01)
        normal_init(self.mask_generate, 0, 0.01)
        
        # normal_init(self.RCNN_deconv, 0, 0.01)
        print('init success!')

if __name__ == '__main__':
    from torch.autograd import Variable
    unet = UNet(3, 2).cuda()
    x = Variable(torch.FloatTensor(1, 3, 1024, 1024)).cuda()
    print(x.size())
    y = unet(x)
    # print(y.size())