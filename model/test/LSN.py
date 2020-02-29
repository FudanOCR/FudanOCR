# -*- coding: utf-8 -*-

def test_LSN(config_file):
    from __future__ import print_function, division
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    import numpy as np
    from lib.model.networkFactory import networkFactory
    from lib.datasets.ctw import CTWDataset, ToTensor
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import os
    from skimage import io,transform,measure,draw
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import scipy.io as sio
    from lib.datasets.proposal_generate import ProposalGenerate
    from lib.utils.connect import generate_proposal
    from lib.model.unet.unet_model import UNet
    import math
    import pickle
    from config import config as config, init_config
    SHOW = True

    def toNp(x):
        return x.data.cpu().numpy()


    def toVar(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    class Demo(object):
        """docstring for Demo"""
        def __init__(self,modelHome,gtPath):
            global SHOW
            super(Demo, self).__init__()
            self.nf = networkFactory(modelHome)
            self.gts = None
            self.gtPath = gtPath
            self.sumProposal = 0
            self.sumHit = 0
            self.sumGt = 0
            self.ratio = 0
            self.show = SHOW
            self.filename = None
            self.savePath = './result'
            self.savepredict = './predict'
            self.testnum = 0
            self.strides = [8,16,32,64]
            self.PG = ProposalGenerate()

        def report(self,image,res):
            #print(res)
            pass

        def saveResult(self,apprContours):
            self.testnum+=1
            print(str(self.testnum)+'/300')
            ans = {}
            ans['accuInf'] = apprContours
            sio.savemat(self.savePath+'/'+self.filename.split('.')[0]+'.mat',ans)

        # def is_rect_overlap(self,rec1,rec2):
        #     nMaxLeft = 0
        #     nMaxTop = 0
        #     nMinRight = 0
        #     nMinBottom = 0
        #     nMaxLeft = np.maximum(rec1[:,0],rec2[:,0])
        #     nMaxTop = np.maximum(rec1[:,1],rec2[:,1])
        #     nMinRight = np.minimum(rec1[:,2],rec2[:,2])
        #     nMinBottom = np.minimum(rec1[:,3],rec2[:,3])
        #     ans = np.ones((len(rec1),len(rec2)))
        #     idx = np.where((nMaxLeft > nMinRight)|nMaxTop > nMinBottom)[0]
        #     ans[:,idx] = 0
        #     return ans

        # def merge_mask_box(self,box1,box2,mask1,mask2):
        #     proposal = box1
        #     proposal[0] = min(box1[0],box2[0])
        #     proposal[1] = min(box1[1],box2[1])
        #     proposal[2] = max(box1[2],box2[2])
        #     proposal[3] = max(box1[3],box2[3])
        #     mask = np.zeros((int(proposal[2]-proposal[0]),int(proposal[3]-proposal[1])))
        #     mask[box1[0]-proposal[0]:box1[2]-proposal[0],box1[1]-proposal[1]:box1[3]-proposal[1]]+=(mask1)
        #     mask[box2[0]-proposal[0]:box2[2]-proposal[0],box2[1]-proposal[1]:box2[3]-proposal[1]]+=(mask2)
        #     cv2.imshow('mask',(mask*125).astype(np.uint8))
        #     cv2.waitKey(0)
        #     return proposal,mask

        # def connect(self,image,pred_mask,bbox,threshold = 0.5):
        #     showimage = image.copy()
        #     proposal_box = []
        #     proposal_mask = []
        #     for idx,box in enumerate(bbox):
        #         if(len(proposal_box)==0):
        #             proposal_box.append(box)
        #             proposal_mask.append(pred_mask[idx]>0.5)
        #             continue
        #         box_overlap = self.is_rect_overlap(np.array([box]),np.array(proposal_box))[0]
        #         box_overlap_idx = np.where(box_overlap>=1)[0]
        #         over_threshold_idx = []
        #         for i in box_overlap_idx:
        #             propposal,mask = self.merge_mask_box(box,proposal_box[i],pred_mask[idx]>0.5,proposal_mask[i])
        #             mask_iou = np.sum(mask>1)/np.sum(mask>0)
        #             if mask_iou>threshold:
        #                 over_threshold_idx.append(i)
        #         proposal = box
        #         mask = pred_mask[idx]>0.5
        #         for j in over_threshold_idx:
        #             proposal,mask = self.merge_mask_box(proposal,proposal_box[j],mask,proposal_mask[j])
        #         for j in over_threshold_idx:
        #             proposal_box.remove(proposal_box[j])
        #             proposal_mask.remove(proposal_mask[j])
        #         proposal_box.append(proposal)
        #         proposal_mask.append(mask)
        #     return proposal_box,proposal_mask



        def display(self,image,res,pred_mask,bbox,sample,circle_labels_pred,all_circle,bbox_score,show=False,threshold=0.5):
        # def display(self,image,circle_labels_pred,all_circle,show=False,threshold=0.4):
            # ==============================================================================
            # mask_all = nn.functional.softmax(pred_mask)
            # score,pred_mask_all = torch.max(mask_all,1)
            # mask_all = mask_all.view(-1,28,28,2)
            # pred_mask_all = torch.max(mask_all,3)[1]
            pred_mask_all = pred_mask.data.cpu().numpy()
            # pred_mask_all = pred_mask.data.cpu().numpy()
            image = image.astype(np.uint8)
            showimage = image.copy()
            showmask = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
            score_list = bbox_score.data.cpu().numpy()
            print(score_list.shape)
            print(pred_mask_all.shape)
            # print(len(bbox))
            savepath = './result/'+config.filename
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            if not os.path.exists(savepath+"/mask"):
                os.makedirs(savepath+"/mask")
            if not os.path.exists(savepath+"/bbox"):
                os.makedirs(savepath+"/bbox")
            if not os.path.exists(savepath+"/score"):
                os.makedirs(savepath+"/score")
            np.save(savepath+"/mask/"+self.filename.split('.')[0]+"_mask.npy", pred_mask_all)
            np.save(savepath+"/bbox/"+self.filename.split('.')[0]+"_bbox.npy", bbox.data.cpu().numpy())
            np.save(savepath+"/score/"+self.filename.split('.')[0]+"_score.npy", score_list)
            # np.save("./result/ratio/"+self.filename.split('.')[0]+"_ratio.npy", )
            # cv2.imwrite("./result/image/"+self.filename,image)

            # anspts = generate_proposal(image,pred_mask_all,bbox.data.cpu().numpy())
            # outputfile = open('./result/predict/'+self.filename.split('.')[0]+'.txt','w')
            # for poly in anspts:
            #     cv2.polylines(image,[poly],True,(0,0,255),3)
            #     # write_poly = poly[:,::-1].reshape(-1)
            #     write_poly = poly.reshape(-1)
            #     print(write_poly)
            #     write_poly = write_poly/self.ratio
            #     write_poly = np.array(write_poly,dtype=np.int32).tolist()
            #     print(write_poly)
            #     write_string = ','.join(str(i) for i in write_poly)
            #     print(write_string)
            #     outputfile.write(write_string+'\n')
            #     cv2.imshow('image',image)
            #     cv2.waitKey(30)
            # cv2.imwrite("./result/disp/"+self.filename,image)
            # outputfile.close()
            # ==============================================================================
            # for idx,box in enumerate(bbox):
            #     score = score_list[idx]
            #     box = box.data.cpu().numpy().astype(np.int32)
            #     # print(score)
            #     # print(box)
            #     if box[0]<0 or box[1]<0:
            #         continue
            #     if box[2]>=showimage.shape[1] or box[3]>=showimage.shape[0]:
            #         continue
            #     if(score<threshold):
            #         continue
            #     cv2.rectangle(showimage,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     mask = np.maximum(pred_mask_all[idx],0)
            #     cv2.imshow('origin_mask',np.array(mask*255,dtype=np.uint8))
            #     # print(mask)
            #     w,h = int(box[2])-int(box[0]),int(box[3])-int(box[1])
            #     # print(box)
            #     # print(mask)
            #     resize_mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)*255
            #     # print(resize_mask)
            #     showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = np.maximum(showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])],resize_mask.astype(np.uint8))
            #     # print(resize_mask)
            #     cv2.imshow('resize_mask',resize_mask.astype(np.uint8))
            # cv2.imshow('showmask',showmask)
            # cv2.imshow('showimage',showimage)
            # cv2.waitKey(0)

            # ==============================================================================
            # image = image.astype(np.uint8)
            # showimage2 = image.copy()
            # cv2.imshow('showimage2',showimage2)
            # cv2.waitKey(0)
            # for stride in self.strides:
            #     circle_labels = nn.functional.softmax(circle_labels_pred[str(stride)])
            #     circle_labels = circle_labels.data.cpu().numpy()
            #     # print(circle_labels)
            #     pos_idx = np.where(circle_labels[:,1]>=0.5)[0]
            #     print(stride,len(pos_idx))
            #     circle = all_circle[str(stride)]
            #     for idx in pos_idx:
            #         box = circle[idx]
            #         cv2.rectangle(showimage2,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            # cv2.imshow('showimage2',showimage2)
            # cv2.waitKey(0)

            # ==============================================================================
            # mask_all = nn.functional.softmax(pred_mask)
            # score,pred_mask_all = torch.max(mask_all,1)
            # mask_all = mask_all.view(-1,14,14,2)
            # pred_mask_all = mask_all[:,:,:,1]>=0.5
            # pred_mask_all = pred_mask_all.view(-1,14,14)
            # image = image.astype(np.uint8)
            # pred_mask = pred_mask.squeeze().data.cpu().numpy().astype(np.float32)
            # bbox = bbox.data.cpu().numpy().astype(np.float32)
            # showmask = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
            # show_point = image.copy()
            # # np.save("./result/mask.npy", pred_mask)
            # # np.save("./result/bbox.npy", bbox)
            # # cv2.imwrite('./result/image.jpg',image)
            # # proposal_image = image.copy()
            # for i,box in enumerate(bbox):
            #     cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,0,255),3)
            #     cv2.circle(show_point,(int((box[0]+box[2])/2),int((box[1]+box[3])/2)),3,(255,0,0),3)
            #     w,h = int(box[2]-box[0]),int(box[3]-box[1])
            #     mask = pred_mask[i]
            #     # print(mask)
            #     resize_mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)*255
            #     showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = (showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] | resize_mask.astype(np.uint8))
            #     # print(np.max(resize_mask))
            #     # print(resize_mask.astype(np.uint8))
            #     # cv2.imshow('region',(mask*255).astype(np.uint8))
            #     # cv2.imshow('mask',resize_mask.astype(np.uint8))
            # # proposal_box,proposal_mask = self.connect(image,pred_mask,bbox)
            # # for proposal in proposal_box:
            # #     cv2.rectangle(proposal_image,(int(proposal[0]),int(proposal[1])),(int(proposal[2]),int(proposal[3])),(0,0,255),3)
            # cv2.imshow('image',image.astype(np.uint8))
            # # cv2.imshow('proposal_image',proposal_image.astype(np.uint8))
            # cv2.imshow('showmask',showmask)
            # cv2.imshow('show_point',show_point)
            # cv2.waitKey(0)


        def rescale(self,image,preferredShort = 768,maxLong = 2048):
            h,w,_ = image.shape
            longSide = max(h,w)
            shortSide = min(h,w)
            self.ratio = preferredShort*1.0/shortSide
            if self.ratio*longSide > maxLong:
                self.ratio = maxLong*1.0/longSide
            image = cv2.resize(image,None,None,self.ratio,self.ratio,interpolation=cv2.INTER_LINEAR)
            return image

        def alignDim(self,image):
            h2,w2,_ = image.shape
            H2 = int(math.ceil(h2/64.0)*64)
            W2 = int(math.ceil(w2/64.0)*64)
            ret_image = np.zeros((H2,W2,_))
            ret_image[:h2,:w2,:] = image
            return ret_image

        def toTensor(self,image):
            # pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
            # image -= pixel_means
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean)/std
            image = image.transpose((2,0,1))
            return torch.from_numpy((image.astype(np.float32)))

        def peerSingleImage(self,image,imageName,display = True,report = True):
            image = self.rescale(image)
            image = self.alignDim(image)
            # cvimage = self.alignDim(cvimage)
            sample = {}
            ptss = [[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]]
            bbx = [[1,1,1,1]]
            all_circle = {}
            for stride in self.strides:
                # print(stride)
                labels=None
                all_anchors = None
                labels,all_anchors,mask_label= self.PG.run(stride,np.array([2,2.5,3,3.5]),[1],image.shape[0]/stride,image.shape[1]/stride,[image.shape[0],image.shape[1],1],0,ptss,image,bbx)
                sample[str(stride)] = all_anchors
                all_circle[str(stride)] = Variable(torch.from_numpy(np.ascontiguousarray(sample[str(stride)].astype(np.float32))).squeeze().cuda(),requires_grad=False)
            tensorimage = image.copy()
            tensor = self.toTensor(tensorimage).unsqueeze(0)
            tensor = Variable(tensor.cuda(),requires_grad = False)
            res = None
            # print(tensor)
            # print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')
            self.model.eval()
            threshold = 0.5
            testMaxNum = 1000
            print(self.model)
            circle_labels_pred,pred_mask,bbox,bbox_idx,pos_idx_stride,bbox_score = self.model.forward(tensor,all_circle,None,istraining=False,threshold=threshold,testMaxNum=testMaxNum) #

            # pred_mask = pred_mask.view(-1,14,14)
            if report:
                self.report(image,res)
            if display:
                self.display(image,res,pred_mask,bbox,sample,circle_labels_pred,all_circle,bbox_score,threshold=threshold)
            torch.cuda.empty_cache()
                # self.display(image,circle_labels_pred,all_circle,threshold=threshold)

        def peerGalary(self,imageFolder,display=True,report = True):
            F = False
            for filename in sorted(os.listdir(imageFolder)):
                print(filename)
                self.filename = filename

                # if not filename == '1072.jpg':
                #     continue
                # if filename == '1342.jpg':
                #     F=True
                # if not F:
                #     continue
                # image = io.imread(os.path.join(imageFolder,filename),plugin = 'pil')
                image = cv2.imread(os.path.join(imageFolder,filename))
                if len(image)==2:
                    image = image[0]
                self.peerSingleImage(image,filename,display,report)

        def prepareNetwork(self,networkPath,type='vgg16'):
            torch.cuda.set_device(1)
            if type == 'vgg16':
                self.savePath = self.savePath+'/'+networkPath.split('/')[-2]+'/'+networkPath.split('/')[-1].split('.')[0]
                # if not os.path.exists(self.savePath):
                #     os.makedirs(self.savePath)
                self.model = self.nf.vgg16()
                pretrainedDict = torch.load(networkPath,map_location='cpu')
                modelDict = self.model.state_dict()
                pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
                modelDict.update(pretrainedDict)
                self.model.load_state_dict(modelDict)
                print('Load model:{}'.format(networkPath))
                self.model.cuda()
                self.model.eval()
            elif type == 'resnet34':
                self.savePath = self.savePath+'/'+networkPath.split('/')[-2]+'/'+networkPath.split('/')[-1].split('.')[0]
                # if not os.path.exists(self.savePath):
                #     os.makedirs(self.savePath)
                self.model = self.nf.resnet34()
                pretrainedDict = torch.load(networkPath,map_location='cpu')
                modelDict = self.model.state_dict()
                pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
                modelDict.update(pretrainedDict)
                self.model.load_state_dict(modelDict)
                print('Load model:{}'.format(networkPath))
                self.model.cuda()
                self.model.eval()
            elif type == 'resnet50':
                self.savePath = self.savePath+'/'+networkPath.split('/')[-2]+'/'+networkPath.split('/')[-1].split('.')[0]
                # if not os.path.exists(self.savePath):
                #     os.makedirs(self.savePath)
                self.model = self.nf.resnet50()
                pretrainedDict = torch.load(networkPath,map_location='cpu')
                modelDict = self.model.state_dict()
                pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
                modelDict.update(pretrainedDict)
                self.model.load_state_dict(modelDict)
                print('Load model:{}'.format(networkPath))
                self.model.cuda()
                self.model.eval()
            elif type == 'unet':
                self.savePath = self.savePath+'/'+networkPath.split('/')[-2]+'/'+networkPath.split('/')[-1].split('.')[0]
                # if not os.path.exists(self.savePath):
                #     os.makedirs(self.savePath)
                self.model = UNet(3,1)
                pretrainedDict = torch.load(networkPath,map_location='cpu')
                modelDict = self.model.state_dict()
                pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
                modelDict.update(pretrainedDict)
                self.model.load_state_dict(modelDict)
                print('Load model:{}'.format(networkPath))
                self.model.cuda()
                self.model.eval()

    demo = Demo('./pretrainmodel','/data/2019AAAI/data/ctw1500/train/text_label_curve')
    demo.prepareNetwork('/data/2019AAAI/output/config013/92.model',type=config.net)
    demo.peerGalary('/data/2019AAAI/data/test',display = True,report = False) #config.testDatasetroot +'/text_image'
    # /home/zhouzhao/Documents/Invoice_test/20170823/ZZSDK
    # /home/zhouzhao/Projects/STD/DataSet/Images/Test

