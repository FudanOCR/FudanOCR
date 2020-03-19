# -*- coding:utf8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model.detection_model.LSN.lib.datasets.generate_anchors import generate_anchors
import time
import cv2
import warnings
import os

warnings.filterwarnings('ignore')

# def mycross(p1,p2,p3): # 叉积判定
#     x1=p2[:,0]-p1[:,0]
#     y1=p2[:,1]-p1[:,1]
#     x2=p3[:,0]-p1[:,0]
#     y2=p3[:,1]-p1[:,1]
#     ans = x1*y2-x2*y1
#     print(ans)
#     return ans

# def segment(p1,p2,p3,p4): #判断两线段是否相交
#     D = np.zeros((len(p1),1))
#     idx = np.where(
#         (max(p1[:,0],p2[:,0])>=min(p3[0],p4[0])) &
#         (max(p3[0],p4[0])>=min(p1[:,0],p2[:,0])) &
#         (max(p1[:,1],p2[:,1])>=min(p3[1],p4[1])) & 
#         (max(p3[1],p4[1])>=min(p1[:,1],p2[:,1])) &
#         (mycross(p1,p2,p3)*mycross(p1,p2,p4)<=0) &
#         (mycross(p3,p4,p1)*mycross(p3,p4,p2)<=0)
#         )[0]
#     D[idx]=1
#     return D
def determinant(v1, v2, v3, v4):
    return (v1*v3-v2*v4)
  
def intersect3(aa, bb, cc, dd):
    D = np.ones((len(aa),1))
    delta = determinant(bb[:,0]-aa[:,0], cc[0]-dd[0], cc[1]-dd[1], bb[:,1]-aa[:,1]);  
    idx = np.where((delta==0))[0]
    D[idx] = 0
    namenda = determinant(cc[0]-aa[:,0], cc[0]-dd[0], cc[1]-dd[1], cc[1]-aa[:,1]) / delta;  
    idx = np.where((namenda>1))
    D[idx] = 0
    idx = np.where((namenda<0))  
    D[idx] = 0
    miu = determinant(bb[:,0]-aa[:,0], cc[0]-aa[:,0], cc[1]-aa[:,1], bb[:,1]-aa[:,1]) / delta;  
    idx = np.where((miu>1))
    D[idx] = 0
    idx = np.where((miu<0))
    D[idx] = 0
    return D

def point_in_polygon(point,polygon,image):
    c = np.zeros((len(point),1))
    i = -1
    l = len(polygon)
    j = l - 1
    while i < l-1:
        i += 1
        temp1 = np.zeros((len(point),1))
        temp2 = np.zeros((len(point),1))
        idx1 = np.where(((polygon[i][0] <= point[:,0]) & (polygon[j][0] > point[:,0])))[0]
        # print(idx1)
        temp1[idx1] = 1
        idx2 = np.where(((polygon[j][0] <= point[:,0]) & (point[:,0] < polygon[i][0])))[0]
        # print(idx2)
        temp1[idx2] = 1
        idx3 = np.where(
            (point[:,1] < ((polygon[j][1] - polygon[i][1]) * (point[:,0] - polygon[i][0]) / (polygon[j][0] - polygon[i][0]) + polygon[i][1]))
        )[0]
        temp2[idx3] = 1
        idx = np.where(
            (temp1==1) & (temp2==1)
        )[0]
        # print(idx)
        c[idx] = 1 - c[idx]
        j = i
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
    return c

class ProposalGenerate(object):

    def __init__(self):
        super(ProposalGenerate, self).__init__()
        # self._allowed_border = allowed_border

    def run(self, feat_stride, scales, ratios ,height, width, im_info, allowed_border, ptss ,image, bbx):
        # print(base_feat)
        self._feat_stride = feat_stride
        self._anchors = generate_anchors(base_size = feat_stride,scales=np.array(scales),ratios=np.array(ratios))
        self._num_anchors = self._anchors.shape[0]
        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = anchors.reshape((K * A, 4))
        templabel = np.zeros((all_anchors.shape[0],1))
        for box in bbx:
            box_h = (box[3]-box[1])
            inds_inside_temp = np.where(
                (all_anchors[:, 0] >= max(0,box[0]-box_h)) &
                (all_anchors[:, 1] >= max(0,box[1]-box_h)) &
                (all_anchors[:, 2] <= min(im_info[1],box[2]+box_h)) &
                (all_anchors[:, 3] <= min(im_info[0],box[3]+box_h)) 
            )[0]
            # print(inds_inside_temp)
            box = list(map(int,box))
            # cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(0,0,255),3)
            templabel[inds_inside_temp] = 1
            # print(inds_inside_temp)
        # print(templabel)
        inds_inside = np.where(templabel==1)[0]
        proposals = all_anchors[inds_inside,:]
        labels = np.zeros((all_anchors.shape[0],1))
        # labels = labels-1
        # labels[inds_inside] = 0
        centerx = (proposals[:,0]+proposals[:,2])/2
        centerx = centerx.reshape(-1,1)
        centery = (proposals[:,1]+proposals[:,3])/2
        centery = centery.reshape(-1,1)
        centerpoint = np.hstack((centerx,centery))
        postive = np.zeros((len(centerpoint),1))
        mask_label = np.zeros((len(labels),28,28))
        kernel = np.ones((3,3), np.uint8)
        
        # print(im_info)
        for pts in ptss:
            mask = np.zeros_like(image).astype(np.uint8)
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            # linecenter = []
            # for i in range(7):
            #     linecenter.append((pts[i]+pts[13-i])/2)
            pts = np.array(pts,dtype = np.int32)
            cv2.fillPoly(mask,[pts],1,1)
            test_mask = mask.copy()
            
            #cv2.imshow('mask1',mask)
            num = 0
            while(np.sum(test_mask)>0):
                test_mask = cv2.erode(test_mask, kernel, iterations=1)
                num+=1
            stide_erode = int(num*1/2)
            test_mask = mask.copy()
            mask = np.array(mask,dtype=np.float32)
            back_ground = test_mask.copy()
            test_mask = cv2.erode(test_mask, kernel, iterations=stide_erode)
            merge_mask = back_ground ^ test_mask
            mask[np.where(merge_mask==1)]=0.1
            
            # back_ground = test_mask.copy()
            # # cv2.imshow('test_mask',np.array(test_mask*255,dtype=np.uint8))
            # # cv2.imshow('back_ground',np.array(back_ground*255,dtype=np.uint8))
            # test_mask = cv2.erode(test_mask, kernel, iterations=stide_erode)
            # merge_mask = back_ground ^ test_mask
            # mask[np.where(merge_mask==1)]=0.6


            # cv2.imshow('mask',np.array(mask*255,dtype=np.uint8))
            # cv2.waitKey(0)
            # uppts = pts[:7].tolist()
            # for i in range(7):
            #     center = [(pts[6-i][0]+pts[7+i][0])*1.0/2,(pts[6-i][1]+pts[7+i][1])*1.0/2]
            #     uppts.append(center)
            # uppts = np.array(uppts,dtype=np.int32)
            # cv2.fillPoly(mask,[uppts],125,1)
            # top_pts = pts[:8]
            # bottom_pts = pts[8:]


            # cv2.imshow('mask',mask)
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            # cv2.polylines(image,[pts],True,(0,0,255),3)
            centerpts = np.zeros_like(pts)
            for i in range(7):
                center = ((pts[i][0]+pts[13-i][0])*1.0/2,(pts[i][1]+pts[13-i][1])*1.0/2)
                # print(center)
                height+=np.sqrt((pts[i][0]-pts[13-i][0])*(pts[i][0]-pts[13-i][0])+(pts[i][1]-pts[13-i][1])*(pts[i][1]-pts[13-i][1]))
                top = ((pts[i][0]-center[0])*2.0/3+center[0],(pts[i][1]-center[1])*2.0/3+center[1])
                bottom = ((pts[13-i][0]-center[0])*2.0/3+center[0],(pts[13-i][1]-center[1])*2.0/3+center[1])
                centerpts[i]=top
                centerpts[13-i]=bottom
            inpolygon = point_in_polygon(centerpoint,pts,image)
            height=height*1.0/7
            top = np.zeros((len(proposals),1))
            bottom = np.zeros((len(proposals),1))
            for i in range(6):
                temptop = intersect3(np.hstack((proposals[:,0].reshape(-1,1),proposals[:,1].reshape(-1,1))),
                np.hstack((proposals[:,2].reshape(-1,1),proposals[:,1].reshape(-1,1))),pts[i],pts[i+1])
                top[np.where(temptop==1)[0]]=1

                temptop = intersect3(np.hstack((proposals[:,2].reshape(-1,1),proposals[:,1].reshape(-1,1))),
                np.hstack((proposals[:,2].reshape(-1,1),proposals[:,3].reshape(-1,1))),pts[i],pts[i+1])
                top[np.where(temptop==1)[0]]=1

                temptop = intersect3(np.hstack((proposals[:,2].reshape(-1,1),proposals[:,3].reshape(-1,1))),
                np.hstack((proposals[:,0].reshape(-1,1),proposals[:,3].reshape(-1,1))),pts[i],pts[i+1])
                top[np.where(temptop==1)[0]]=1

                temptop = intersect3(np.hstack((proposals[:,0].reshape(-1,1),proposals[:,3].reshape(-1,1))),
                np.hstack((proposals[:,0].reshape(-1,1),proposals[:,1].reshape(-1,1))),pts[i],pts[i+1])
                top[np.where(temptop==1)[0]]=1
                # print(np.where(top==1)[0])
                tempbottom = intersect3(np.hstack((proposals[:,0].reshape(-1,1),proposals[:,1].reshape(-1,1))),
                np.hstack((proposals[:,2].reshape(-1,1),proposals[:,1].reshape(-1,1))),pts[13-i],pts[12-i])
                bottom[np.where(tempbottom==1)[0]]=1

                tempbottom = intersect3(np.hstack((proposals[:,2].reshape(-1,1),proposals[:,1].reshape(-1,1))),
                np.hstack((proposals[:,2].reshape(-1,1),proposals[:,3].reshape(-1,1))),pts[13-i],pts[12-i])
                bottom[np.where(tempbottom==1)[0]]=1

                tempbottom = intersect3(np.hstack((proposals[:,2].reshape(-1,1),proposals[:,3].reshape(-1,1))),
                np.hstack((proposals[:,0].reshape(-1,1),proposals[:,3].reshape(-1,1))),pts[13-i],pts[12-i])
                bottom[np.where(tempbottom==1)[0]]=1

                tempbottom = intersect3(np.hstack((proposals[:,0].reshape(-1,1),proposals[:,3].reshape(-1,1))),
                np.hstack((proposals[:,0].reshape(-1,1),proposals[:,1].reshape(-1,1))),pts[13-i],pts[12-i])
                bottom[np.where(tempbottom==1)[0]]=1
                # print(np.where(bottom==1)[0])

            # print(np.where(c==1)[0])
            anchorheight = np.minimum(proposals[:,2]-proposals[:,0],proposals[:,3]-proposals[:,1])
            h = np.zeros((len(proposals),1))
            h[np.where((anchorheight/height<=1.8))[0]] = 1
            postive_idx = np.where(
                (inpolygon==1) & (h==1) & (bottom==1) & (top ==1)
            )[0]
            postive[postive_idx] = 1
            # showimage = image.copy()
            for idx in postive_idx: # [np.where(postive==1)[0]]
                an = proposals[idx]
                an = list(map(int,an))
                # showimage = image.copy()
                anchor_mask = mask[an[1]:an[3],an[0]:an[2]]
                mask_label[inds_inside[idx],:,:] = cv2.resize(anchor_mask,(28,28),interpolation=cv2.INTER_AREA)

                # cv2.imshow('anchor_mask',mask_label[idx,:,:])
                # cv2.rectangle(showimage,(an[0],an[1]),(an[2],an[3]),(0,255,0),3)
                # cv2.imshow('image',showimage)
                # cv2.waitKey(0)
            # print(np.where(top==1)[0])
            # cv2.imshow('image',image)
            # cv2.waitKey(0)
            # print(c)
        # print()
        # print(len(np.where(idxin==1)[0]))
        labels[inds_inside] = postive
        return labels,all_anchors,mask_label
        
# def dist(pts):
#     ret = 0.0
#     for i in range(pts.shape[0]-1):
#         ret += np.linalg.norm(pts[i]-pts[i+1])
#     return ret

# def rot(pts):
#     ptsnew = []
#     ptsnew.append(pts[6])
#     begin = pts[6]
#     end = pts[7]
#     duration = (end - begin)/6.0
#     for i in range(1,6,1):
#         ptsnew.append((begin+duration*i).astype(np.int))
#     ptsnew.append(pts[7])
#     ptsnew.append(pts[13])
#     begin = pts[13]
#     end = pts[0]
#     duration = (end - begin)/6.0
#     for i in range(1,6,1):
#         ptsnew.append((begin+duration*i).astype(np.int))
#     ptsnew.append(pts[0])
#     return np.array(ptsnew,dtype=np.float32)

# def loadData(sample):
#     image = cv2.imread(sample[0])
#     # image= io.imread(sample[0], plugin='pil')
#     # canvas = image.copy()
#     labelfile = open(sample[1])
#     lines = labelfile.readlines()
#     labelfile.close()
#     bbx = []
#     ptss = []
#     for line in lines:
#         points = line.strip().split(',')
#         points = list(map(int,points))
#         xmin,ymin,xmax,ymax = (points[0],points[1],points[2],points[3])
#         pts = np.array(points[4:],dtype = np.float32).reshape(-1,2)
#         pts[:,0]+=xmin
#         pts[:,1]+=ymin
#         avglong = dist(pts[0:7,:]) + dist(pts[7:14,:])
#         avgshort = dist(pts[6:8,:]) + dist(np.vstack(([pts[13,:],pts[0,:]])))
#         if avglong < avgshort:
#             pts = rot(pts)
#         bbx.append([xmin,ymin,xmax,ymax])
#         ptss.append(pts)
#     ptss = np.ascontiguousarray(ptss,np.float32)
#     bbx = np.ascontiguousarray(bbx,dtype=np.float32)
#     return image,ptss,bbx

# if __name__ == '__main__':
#     datasetroot = '/data/2019AAAI/data/ctw1500/train'
#     txtpath = '/data/2019AAAI/data/ctw1500/train/gt_box'
#     if not os.path.exists(txtpath):
#         os.makedirs(txtpath)
#     imagelist = os.listdir(datasetroot+'/text_image')
#     PG = ProposalGenerate()
#     samplelist = []
#     strides = [8,16,32,64,128]
#     for imagename in imagelist:
#         samplelist.append([datasetroot+'/text_image/'+imagename,datasetroot+'/text_label_curve/'+imagename.split('.')[0]+'.txt'])
#     for sample in samplelist:
#         image,ptss,bbx = loadData(sample)
#         image_name = sample[0].split('/')[-1]
#         box_file = open(os.path.join(txtpath,image_name.split('.')[0]+'.txt'),'w')
#         showimage = image.copy()
#         for stride in strides:
#             labels,all_anchors,mask_label = PG.run(stride,np.array([2,2.5,3,3.5]),[1],image.shape[0]//stride,image.shape[1]//stride,[image.shape[0],image.shape[1],1],0,ptss,image,bbx)
#             for idx in np.where(labels==1)[0]: #
#                 an = all_anchors[idx]
#                 box_file.write(str(an[0])+' '+str(an[1])+' '+str(an[2])+' '+str(an[3])+'\n')
#                 cv2.rectangle(showimage,(an[0],an[1]),(an[2],an[3]),(0,255,0),3)
#         cv2.imshow('showimage',showimage)
#         cv2.waitKey(0)
#         box_file.close()

    