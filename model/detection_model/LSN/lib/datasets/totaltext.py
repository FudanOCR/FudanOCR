from __future__ import absolute_import
from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import os
from skimage import io, transform
import random
import scipy.io as sio
import cv2
import sys
# sys.path.insert(0,'./')
# import config
from lib.datasets.proposal_generate_totaltext import ProposalGenerate
import math
from imgaug import augmenters as iaa


def dist(pts):
    ret = 0.0
    for i in range(pts.shape[0]-1):
        ret += np.linalg.norm(pts[i]-pts[i+1])
    return ret

class Flip_image(object):
    
    def __init__(self):
        pass

    def __call__(self,image,ptss,bbx):
        k = random.randint(0,2)
        if k==0:
            return image,ptss,bbx
        elif k==1:
            h,w,_ = image.shape
            retimage = image[:,::-1,:]
            ptss[:,:,0] = w-ptss[:,:,0]
            # ptss[:,:,1] = w-ptss[:,:,1]
            ret_bbx = bbx.copy()
            ret_bbx[:,0] = w-bbx[:,2]
            # bbx[:,1] = w-bbx[:,1]
            ret_bbx[:,2] = w-bbx[:,0]
            # bbx[:,3] = w-bbx[:,3]
            return retimage, ptss, ret_bbx
        else:
            h,w,_ = image.shape
            retimage = image[::-1,:,:]
            ptss[:,:,1] = h-ptss[:,:,1]
            # ptss[:,:,1] = w-ptss[:,:,1]
            ret_bbx = bbx.copy()
            ret_bbx[:,1] = h-bbx[:,3]
            # bbx[:,1] = w-bbx[:,1]
            ret_bbx[:,3] = h-bbx[:,1]
            # bbx[:,3] = w-bbx[:,3]
            return retimage, ptss, ret_bbx

class Rotate(object):
    def __init__(self, angleList):
        self.angleList = angleList

    def __call__(self, image,ptss,bbx):

        # angle = random.randint(config.rotate_angle[0],config.rotate_angle[-1])
        # # print(angle)
        angle = self.angleList[random.randint(0, len(self.angleList)-1)]
        if(angle % 360 == 0):
            return image,ptss,bbx
        h, w, c = image.shape

        _cos = np.cos(angle/180.0*3.1416)
        _sin = np.sin(angle/180.0*3.1416)

        M1 = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]])
        M2 = np.array([[_cos, _sin, 0], [-_sin, _cos, 0], [0, 0, 1]])

        tmp = np.dot(np.dot(M2, M1), np.array(
            [[0, 0, w, w], [0, h, h, 0], [1, 1, 1, 1]]))

        left = np.floor(np.min(tmp[0]))
        top = np.floor(np.min(tmp[1]))
        right = np.ceil(np.max(tmp[0]))
        bottom = np.ceil(np.max(tmp[1]))
        new_w = right - left + 1
        new_h = bottom - top + 1

        M3 = np.array([[1, 0, new_w/2], [0, 1, new_h/2], [0, 0, 1]])
        M = np.dot(M3, np.dot(M2, M1))
        for i in range(ptss.shape[0]):
            pts = np.hstack((ptss[i, :, :], np.ones((len(ptss[i, :, :]), 1))))
            pts = np.dot(M, pts.T)[0:2].T
            bbx[i, 0] = np.min(pts[:, 0])
            bbx[i, 1] = np.min(pts[:, 1])
            bbx[i, 2] = np.max(pts[:, 0])
            bbx[i, 3] = np.max(pts[:, 1])

            ptss[i, :, :] = pts

        retimage = cv2.warpAffine(
            image, M[0:2], (int(new_w), int(new_h)))  # ???
        # sample['image'] = retimage
        # sample['ptss'] = ptss
        # sample['bbx'] = bbx
        return retimage,ptss,bbx

class ChangeRatio(object):
    def __init__(self,ratiolist):
        self.ratiolist = ratiolist

    def __call__(self, image,ptss,bbx):
        wratio = self.ratiolist[random.randint(0, len(self.ratiolist)-1)]
        hratio = self.ratiolist[random.randint(0, len(self.ratiolist)-1)]
        retimage = cv2.resize(image,None,None,wratio,hratio)
        # print(ptss,bbx)
        ptss[:,:,0] = ptss[:,:,0]*wratio
        ptss[:,:,1] = ptss[:,:,1]*hratio
        bbx[:,0] = bbx[:,0]*wratio
        bbx[:,1] = bbx[:,1]*hratio
        bbx[:,2] = bbx[:,2]*wratio
        bbx[:,3] = bbx[:,3]*hratio
        return retimage,ptss,bbx

class RandomCrop(object):
    def __init__(self):
        pass

    def __call__(self, image,ptss,bbx):
        sidex = random.randint(0, 2)
        sidey = random.randint(0, 2)
        # print(bbx)
        left = 0
        top = 0
        right = image.shape[1]
        bottom = image.shape[0]
        if sidex == 1:
            left = np.min(bbx[:,0])
        elif sidex == 2:
            right = np.max(bbx[:,2])
        if sidey == 1:
            top = np.min(bbx[:,1])
        elif sidey == 2:
            bottom = np.max(bbx[:,3])
        left = int(left)
        right = int(right)
        top = int(top)
        bottom = int(bottom)
        image = image[top:bottom,left:right,:]
        bbx[:,0]-=left
        bbx[:,2]-=left
        bbx[:,1]-=top
        bbx[:,3]-=top
        ptss[:,:,0]-=left
        ptss[:,:,1]-=top
        return image,ptss,bbx
        

class ToTensor(object):

    def __init__(self,strides):
        self.strides = strides

    def __call__(self, sample):
        image = sample['image']
        image = image.astype(np.float32,copy=False)
        h2, w2, _ = image.shape
        # pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean)/std
        # image -= pixel_means
        image = image.transpose((2, 0, 1))
        ret_sample = {}
        ret_sample['image'] = torch.from_numpy((image.astype(np.float32))) ##????
        # ret_sample['mask'] = torch.from_numpy((sample['mask'].astype(np.float32))).unsqueeze(0)
        # ret_sample['resize_mask'] = torch.from_numpy((sample['resize_mask'].astype(np.float32))).unsqueeze(0)
        for stride in self.strides:
            ret_sample['labels_stride_'+str(stride)] = torch.from_numpy(
                np.ascontiguousarray(sample['labels_stride_'+str(stride)].astype(np.int))).squeeze()
            ret_sample['anchors_stride_'+str(stride)] = sample['anchors_stride_'+str(stride)].astype(np.float32)
            ret_sample['mask_stride_'+str(stride)] = torch.from_numpy((sample['mask_stride_'+str(stride)].astype(np.float32))).squeeze()
            # print(ret_sample['mask_stride_'+str(stride)].size())
        return ret_sample

def loadData(sample):
    image = cv2.imread(sample[0])
    mat_contents = sio.loadmat(sample[1])
    boxes = mat_contents['polygt']
    bbx = []
    ptss = []
    ptss_idx = []
    for box in boxes:
        if len(box[4]) == 0:
            continue
        if box[4][0] == '#':
            continue
        x = np.array(box[1],dtype=np.float32)
        y = np.array(box[3],dtype=np.float32)
        pts = np.ascontiguousarray(np.hstack((x.reshape(-1,1),y.reshape(-1,1))))
        xmin = np.min(pts[:,0])
        ymin = np.min(pts[:,1])
        xmax = np.max(pts[:,0])
        ymax = np.max(pts[:,1])
        bbx.append([xmin,ymin,xmax,ymax])
        ptss.append(pts)
        ptss_idx.append(len(pts))
    ptss_idx = np.ascontiguousarray(ptss_idx,dtype=np.int32)
    max_len = np.max(ptss_idx)
    padding_ptss = []
    for pts in ptss:
        padding_len = max_len-len(pts)
        padding_array = np.zeros((padding_len,2))
        padding_pts = np.vstack((pts,padding_array))
        padding_ptss.append(padding_pts)
    ptss = np.ascontiguousarray(padding_ptss,dtype=np.float32)
    bbx = np.ascontiguousarray(bbx,dtype=np.float32)
    return image,ptss,bbx,ptss_idx

def resize(image,ptss,bbx,preferredShort,maxLong):
    h,w,c = image.shape
    shortSide,longSide = min(h,w),max(h,w)
    ratio = preferredShort*1.0/shortSide
    if longSide*ratio>maxLong:
        ratio = maxLong*1.0/longSide
    retimage = cv2.resize(image,None,None,ratio,ratio,interpolation=cv2.INTER_LINEAR)
    ptss*=ratio
    bbx*=ratio
    return retimage,ptss,bbx

class TotalTextDataset(Dataset):
    def __init__(self, datasetroot,transforms,strides,istraining = True,data='synthtext'):
        self.istraining = istraining
        self.datasetroot = datasetroot
        self.transforms = transforms
        # imagelist = os.listdir(datasetroot+'/text_image')
        self.data = data
        self.preferredShortlist = np.arange(512,1280,64)
        angle = np.arange(0,360,30)
        self.rotate_data = Rotate(angle)
        self.change_ratio = ChangeRatio(np.array([0.5,1,1.5]))
        self.PG = ProposalGenerate()
        self.strides = strides
        self.random_crop = RandomCrop()
        self.samplelist = []
        self.filp_image = Flip_image()
        if data == 'totaltext':
            for gtname in os.listdir(os.path.join(datasetroot,'gt')):
                #gtname = 'poly_gt_img490.mat'
                self.samplelist.append([os.path.join(datasetroot,'image',gtname.split('.')[0][8:]+'.jpg'), os.path.join(datasetroot,'gt',gtname)])
                

    def __len__(self):
        return len(self.samplelist)

    def __getitem__(self, idx):
        try:
            imageinfo = self.samplelist[idx]
            self.image_name = imageinfo[0].split('/')[-1]
            image,ptss,bbx,ptss_idx = loadData(imageinfo)
            if self.data == 'totaltext':
                if self.istraining:
                    preferredShort = self.preferredShortlist[random.randint(0,len(self.preferredShortlist)-1)]
                    image,ptss,bbx = self.rotate_data(image,ptss,bbx)
                    image,ptss,bbx = self.change_ratio(image,ptss,bbx)
                    image,ptss,bbx = self.random_crop(image,ptss,bbx)
                    image,ptss,bbx = self.filp_image(image,ptss,bbx)
                    seg = iaa.Sequential([iaa.Invert(random.randint(0,100)*1.0/100),iaa.GaussianBlur(random.randint(0,10)*1.0/10)],random_order=True)
                    image = seg.augment_images([image])[0]
                else:
                    preferredShort = 640

            image,ptss,bbx = resize(image,ptss,bbx,preferredShort,1664)

            padimage = np.zeros((int(math.ceil(image.shape[0]*1.0/128)*128),int(math.ceil(image.shape[1]*1.0/128)*128),3),dtype = np.uint8)
            padimage[0:image.shape[0],0:image.shape[1],:] = image
            sample = {}
            for stride in self.strides:
                labels=None
                all_anchors = None
                labels,all_anchors,mask_label = self.PG.run(stride,np.array([2,2.5,3,3.5]),[1],padimage.shape[0]/stride,padimage.shape[1]/stride,[padimage.shape[0],padimage.shape[1],1],0,ptss,padimage,bbx,ptss_idx)
                sample['labels_stride_'+str(stride)] = labels
                mask_label = (mask_label).astype(np.float32)
                sample['mask_stride_'+str(stride)] = mask_label
                sample['anchors_stride_'+str(stride)] = all_anchors
            sample['image'] = padimage
            if self.transforms:
                sample = self.transforms(sample)
            sample['imagename'] = imageinfo[0].split('/')[-1] 
            return sample
        except:
            return "FAIL"
            


if __name__ == '__main__':
    datasetroot = '/data/2019AAAI/data/total-text/train'
    # savepath = '/data/2019AAAI/disp/'
    strides = [8,16,32,64]
    dataTransform = transforms.Compose([ToTensor(strides)])
    CtwDataset = TotalTextDataset(datasetroot, dataTransform,strides,istraining=True,data='totaltext')
    dataloader = DataLoader(CtwDataset, batch_size=1,shuffle=False, num_workers=5)
    # txtpath = '/data/2019AAAI/data/ctw1500/train/gt_box'
    # if not os.path.exists(txtpath):
    #     os.makedirs(txtpath)
    # while True:
    num = 0
    for sample in dataloader:
        # print(num)
        # if num != 2:
        #     num+=1
        #     continue
        image = sample['image']
        image_name = sample['imagename'][0]
        # print(image_name)
        # print(image)
        print(image_name)
        #if not image_name=='img490.jpg':
        #    continue
        image = image.squeeze().numpy().transpose((1, 2, 0))
        # pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        # image = image + pixel_means
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image*std+mean)
        image = image.astype(np.uint8,copy=False)
        showimage = image.copy()
        # box_file = open(os.path.join(txtpath,image_name.split('.')[0]+'.txt'),'w')
        # print(sample)
        for stride in strides:
            # print(stride)
            labels = sample['labels_stride_'+str(stride)].squeeze().numpy()
            all_anchors = sample['anchors_stride_'+str(stride)][0].numpy()
            mask_list = sample['mask_stride_'+str(stride)].squeeze().numpy()
            # print(mask_list.shape)
            # print(all_anchors)
            # box_file.write(str(len(np.where(labels==1)[0]))+'\n')
            for idx in np.where(labels==1)[0]: #
                an = all_anchors[idx]
                # box_file.write(str(an[0])+' '+str(an[1])+' '+str(an[2])+' '+str(an[3])+'\n')
                an = list(map(int,an))
                anchor_mask = mask_list[idx,:,:]*255
                # print(anchor_mask)
                # print(an)
                # showimage = image.copy()
                cv2.rectangle(showimage,(an[0],an[1]),(an[2],an[3]),(0,255,0),3)
            # print(len(labels))
                cv2.imshow('anchor_mask',anchor_mask.astype(np.uint8))
        cv2.imshow('image',showimage)
        cv2.waitKey(30)
        # cv2.imwrite(savepath+str(num)+'.jpg',showimage)
        num+=1
        # box_file.close()
            
