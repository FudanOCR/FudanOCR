import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from operator import itemgetter
import math
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



DEBUG = False

princurve = importr('princurve',on_conflict="warn")
rpy2.robjects.numpy2ri.activate()

def point_in_polygon(point,polygon):
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
    return c

def rotate_image(image,angle,point,ptss):
    h, w = image.shape
    print(h,w)
    H = max(h,w)
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    M1 = np.array([[1, 0, -point[0]], [0, 1, -point[1]], [0, 0, 1]])
    M2 = np.array([[_cos, _sin, 0], [-_sin, _cos, 0], [0, 0, 1]])

    tmp = np.dot(np.dot(M2, M1), np.array(
        [[0, 0, H, H], [0, H, H, 0], [1, 1, 1, 1]]))

    left = np.floor(np.min(tmp[0]))
    top = np.floor(np.min(tmp[1]))
    right = np.ceil(np.max(tmp[0]))
    bottom = np.ceil(np.max(tmp[1]))
    new_w = right - left + 1
    new_h = bottom - top + 1

    M3 = np.array([[1, 0, point[0]], [0, 1, point[1]], [0, 0, 1]])
    M = np.dot(M3, np.dot(M2, M1))

    pts = np.hstack((ptss[:, :], np.ones((ptss.shape[0], 1))))
    pts = np.dot(M, pts.T)[0:2].T

    ptss[:, :] = pts

    retimage = cv2.warpAffine(
        image, M[0:2], (int(new_w), int(new_h)))  # ???

    return retimage,M,ptss

def is_rect_overlap(rec1,rec2):
    nMaxLeft = 0
    nMaxTop = 0
    nMinRight = 0
    nMinBottom = 0
    nMaxLeft = np.maximum(rec1[:,0],rec2[:,0])
    nMaxTop = np.maximum(rec1[:,1],rec2[:,1])
    nMinRight = np.minimum(rec1[:,2],rec2[:,2])
    nMinBottom = np.minimum(rec1[:,3],rec2[:,3])
    ans = np.ones((len(rec1),len(rec2)))
    idx = np.where((nMaxLeft > nMinRight)|(nMaxTop > nMinBottom))[0]
    ans[:,idx] = 0
    return ans

def merge_mask_box(box1,box2,mask1,mask2):
    proposal = box1.copy()
    proposal[0] = min(box1[0],box2[0])
    proposal[1] = min(box1[1],box2[1])
    proposal[2] = max(box1[2],box2[2])
    proposal[3] = max(box1[3],box2[3])
    mask = np.zeros((int(proposal[3]-proposal[1]),int(proposal[2]-proposal[0])))
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)
    # print(mask1.shape)
    # print(mask2.shape)
    # print(mask[box1[1]-proposal[1]:box1[3]-proposal[1],box1[0]-proposal[0]:box1[2]-proposal[0]].shape)
    # print(mask[box2[1]-proposal[1]:box2[3]-proposal[1],box2[0]-proposal[0]:box2[2]-proposal[0]].shape)
    # print(box2[1]-proposal[1],box2[3]-proposal[1],box2[0]-proposal[0],box2[2]-proposal[0])
    # print(mask.shape)
    # print(box1,box2,proposal)
    mask[box1[1]-proposal[1]:box1[3]-proposal[1],box1[0]-proposal[0]:box1[2]-proposal[0]] = np.maximum(mask[box1[1]-proposal[1]:box1[3]-proposal[1],box1[0]-proposal[0]:box1[2]-proposal[0]],mask1)
    mask[box2[1]-proposal[1]:box2[3]-proposal[1],box2[0]-proposal[0]:box2[2]-proposal[0]] = np.maximum(mask2,mask[box2[1]-proposal[1]:box2[3]-proposal[1],box2[0]-proposal[0]:box2[2]-proposal[0]])

    # cv2.imshow('mask',(mask*255).astype(np.uint8))
    # cv2.waitKey(0)
    return proposal,mask

def distance_point(point1,point2):
    return np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))

def connect(image,pred_mask,bbox,threshold = 0.4):
    bbox = np.array(bbox,dtype=np.int32)
    showimage = image.copy()
    proposal_box = []
    proposal_mask = []
    showmask = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
    for idx,box in enumerate(bbox):
        w,h = int(box[2])-int(box[0]),int(box[3])-int(box[1])
        
        if box[0]<0 or box[1]<0 or box[2]>showmask.shape[1] or box[3]>showmask.shape[0]:
            continue
        resize_mask = cv2.resize(pred_mask[idx],(w,h),interpolation=cv2.INTER_NEAREST)
        erode_num = 0
        kernel = np.ones((3,3), np.uint8)
        # print(resize_mask)
        ## cv2 erode and dilate
        # print(w/8)
        resize_mask = np.array(resize_mask>=0.3,dtype=np.uint8) ### 
        while(True):
            erode_num +=1
            resize_mask = cv2.erode(resize_mask, kernel, iterations=1)
            im2, contours, hierarchy = cv2.findContours(resize_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            # resize_mask1 = np.array((resize_mask==1),dtype=np.uint8)
            # resize_mask2 = np.array((resize_mask==2),dtype=np.uint8)
            # im21, contours1, hierarchy1 = cv2.findContours(resize_mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # im22, contours2, hierarchy2 = cv2.findContours(resize_mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow('resize_mask',np.array(resize_mask*255,dtype=np.uint8))
            # cv2.waitKey(0)
            if len(contours) <= 1:
                break
        erode_num -=1
        while(erode_num):
            erode_num-=1
            resize_mask = cv2.dilate(resize_mask,kernel, iterations=1)
        mask = resize_mask.copy()
        # cv2.imshow('mask',np.array(mask*255,dtype=np.uint8))
        # cv2.waitKey(0)
        # mask = np.array(mask>0,dtype=np.uint8)
        if np.sum(mask>0)*1.0/(mask.shape[0]*mask.shape[1]) < 0.2:
            continue
        if np.sum(mask==1) == 0:
            continue
        if(len(proposal_box)==0):
            proposal_box.append(box.tolist())
            proposal_mask.append(mask.tolist())
            continue
        box_overlap = is_rect_overlap(np.array([box]),np.array(proposal_box))[0]
        box_overlap_idx = np.where(box_overlap>=1)[0]
        over_threshold_idx = []
        showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = np.maximum(showmask[int(box[1]):int(box[3]),int(box[0]):int(box[2])],(np.array(mask)*255).astype(np.uint8))
        # cv2.imshow('showmask',(np.array(showmask)).astype(np.uint8))
        # print('start_merge_box_proposal')
        # print(proposal_box)
        # print(proposal_mask)
        for i in box_overlap_idx:
            propposal,merge_mask = merge_mask_box(box,proposal_box[i],mask,proposal_mask[i])
            proposal_mask_sample = np.array(proposal_mask[i]) 
            mask_iou = (np.sum(mask==1) + np.sum(proposal_mask_sample==1) - np.sum(merge_mask==1))/ min(np.sum(mask==1),np.sum(proposal_mask_sample==1))
            # print(mask_iou)
            # print('mask_iou:',mask_iou)
            # cv2.imshow('mask',(np.array(mask)*250).astype(np.uint8))
            # cv2.imshow('proposal_mask',(np.array(proposal_mask[i])*250).astype(np.uint8))
            # cv2.imshow('merge_mask',(np.array(merge_mask)*250).astype(np.uint8))
            # cv2.waitKey(0)
            if mask_iou>=threshold:
                over_threshold_idx.append(i)
        proposal = box
        deleteindex = []
        # print("start_merge_proposal")
        # print(len(proposal_box))
        for j in over_threshold_idx:
            proposal,mask = merge_mask_box(proposal,proposal_box[j],mask,proposal_mask[j])
            deleteindex.append(j)
        deleteindex.reverse()
        for idx in deleteindex:
            del proposal_box[idx]
            del proposal_mask[idx]
        proposal_box.append(proposal.tolist())
        proposal_mask.append(mask.tolist())
        # print('\n======================================================================================\n')
        # for i in range(len(proposal_box)):
        #     print(np.array(proposal_mask[i]).shape)
        #     print(proposal_box[i][3]-proposal_box[i][1],proposal_box[i][2]-proposal_box[i][0])
        # print('\n======================================================================================\n')
        # print(len(proposal_box))
        
        # cv2.imshow('showmask',showmask)
        # cv2.waitKey(0)
    return proposal_box,proposal_mask


def generate_proposal(image,pred_mask,bbox):
    proposal_box,proposal_mask = connect(image,pred_mask,bbox)

    # proposal_box,proposal_mask = connect(image,np.array(proposal_mask),np.array(proposal_box))
    ansbox = []
    for idx,mask in enumerate(proposal_mask):
        mask = np.array(mask,dtype=np.uint8)
        isreverse = False
        if mask.shape[0]/mask.shape[1]>1.2:
            mask = np.swapaxes(mask,0,1)
            isreverse = True
        kernel = np.ones((3,3), np.uint8)
        test_mask = mask.copy()
        erode_num = 0
        while(erode_num<=10):
            erode_num +=1
            test_mask = cv2.erode(test_mask, kernel, iterations=1)
            im2, contours, hierarchy = cv2.findContours(test_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow('mask',(np.array(test_mask)*255).astype(np.uint8))
            # cv2.waitKey(0)
            if len(contours) <= 1:
                break
        while(erode_num):
            erode_num-=1
            test_mask = cv2.dilate(test_mask,kernel, iterations=1)
            # cv2.imshow('mask',(np.array(test_mask)*125).astype(np.uint8))
            # cv2.waitKey(0)
        
        mask = test_mask.copy()
        # print('cent')
        centy,centx = np.where(np.array(mask)>=1)
        if centy.shape[0]<=1 or centx.shape[0]<=1:
            continue
        sorted_idx = np.argsort(centx)
        centx = centx[sorted_idx]
        centy = centy[sorted_idx]
        centpts = np.array((centx,centy)).T
        num = centpts.shape[0]
        stride = num//200
        stride = max(1,stride)
        centidx = np.arange(0,num,stride) 
        centnewpts = centpts[centidx,:]
        if len(centnewpts)<5:
            continue

        subimage = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        subimage3 = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        # for i in range(len(centnewpts)):
        #     cv2.circle(subimage3,tuple(centnewpts[i]),1,(0,255,0),1)
        # cv2.imshow('subimage3',subimage3)
        # cv2.waitKey(0)

        # subimage = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        # for i in range(len(centcentpts)-1):
        #     cv2.line(subimage,tuple(centcentpts[i]),tuple(centcentpts[i+1]),(0,255,0),3)
        # cv2.imshow('subimage',subimage)
        # cv2.waitKey(0)
        subimage2 = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        nr,nc = centnewpts.shape
        try:
            centnewpts = ro.r.matrix(centnewpts, nrow=nr, ncol=nc)
            centres = princurve.principal_curve(centnewpts,thresh=0.1)[0] #
            centcentpts = np.array(centres,dtype=np.int32)
        except:
            continue
        # if DEBUG:
        #     for i in range(len(centcentpts)-1):
        #         cv2.circle(subimage2,tuple(centcentpts[i]),1,(0,255,0),1)
        #         cv2.imshow('subimage2',cv2.resize(subimage2,None,None,2,2))
                # cv2.waitKey(0)
        cent_idx = np.linspace(0,centcentpts.shape[0]-1,7).astype(np.int32)
        # centcentpts = np.array(sorted(centcentpts,key=itemgetter(0,1)))
        centcentpts = centcentpts[cent_idx]

        if isreverse:
            centcentpts = centcentpts[:,::-1]
            mask = np.swapaxes(mask,0,1)

        
        
        

        # print('\n==================\n')
        # print(centcentpts)
        # centcentpts = ClockwiseSortPoints(centcentpts)
        # print(centcentpts)
        # print('\n==================\n')

        if DEBUG:
            for i in range(len(centcentpts)-1):
                cv2.line(subimage,tuple(centcentpts[i]),tuple(centcentpts[i+1]),(0,255,0),3)
                cv2.imshow('subimage',cv2.resize(subimage,None,None,2,2))
                cv2.waitKey(0)

        proposal = proposal_box[idx]
        # centcentpts[:,1] += proposal[1]
        # centcentpts[:,0] += proposal[0]

        ## 删除重复的中心点
        idx = len(centcentpts)-2
        while(idx>=0):
            if centcentpts[idx][0]==centcentpts[idx+1][0] and centcentpts[idx][1]==centcentpts[idx+1][1]:
                centcentpts = np.delete(centcentpts,idx+1,axis=0)
            idx-=1 

        # if len(centcentpts)==1:
        #     sumdist = 0
        #     for i in range(7):
        #         sumdist+=distance_point(downcentpts[i],upcentpts[i])
        #     dist = sumdist/7
        #     subbox = []
        #     subbox.append([centcentpts[0]-dist,centcentpts[1]-dist])
        #     subbox.append([centcentpts[0]+dist,centcentpts[1]-dist])
        #     subbox.append([centcentpts[0]-dist,centcentpts[1]+dist])
        #     subbox.append([centcentpts[0]+dist,centcentpts[1]+dist])
        #     continue

        ## 删除不符合的中心点
        newcentcentpts = []
        newcentcentpts.append(centcentpts[0])
        for i in range(1,len(centcentpts)-1):
            vectera = centcentpts[i]-centcentpts[i-1]
            vecterb = centcentpts[i+1]-centcentpts[i]
            
            cosab = (vectera[0]*vecterb[0]+vectera[1]*vecterb[1])/(np.sqrt(vectera[0]*vectera[0]+vectera[1]*vectera[1])+np.sqrt(vecterb[0]*vecterb[0]+vecterb[1]*vecterb[1]))
            if cosab < 0:
                # centcentpts = [centcentpts[0],centcentpts[-1]]
                print('break',cosab)
            else:
                newcentcentpts.append(centcentpts[i])
                # break
        newcentcentpts.append(centcentpts[-1])
        # newcentcentpts = []
        # newcentcentpts.append(centcentpts[0])
        # for i in range(1,len(centcentpts)):
        #     vecterb = newcentcentpts[len(newcentcentpts)-1]-centcentpts[i]
        #     Len = np.sqrt(vecterb[1]*vecterb[1]+vecterb[0]*vecterb[0])
        #     if Len>=10 or i==len(centcentpts)-1:
        #         newcentcentpts.append(centcentpts[i])
        
        centcentpts = np.array(newcentcentpts)
        # print('centcentpts',centcentpts)

        # draw_ptss = centcentpts.copy()
        # draw_ptss[:,1]+=proposal[1]
        # draw_ptss[:,0]+=proposal[0]  
        # subimage = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        # for i in range(len(centcentpts)-1):
        #     cv2.line(subimage,tuple(centcentpts[i]),tuple(centcentpts[i+1]),(0,255,0),3)
        # for i in range(len(draw_ptss)-1):
        #     cv2.line(image,tuple(draw_ptss[i]),tuple(draw_ptss[i+1]),(0,0,255),3)

        ## 根据中心线算出proposal
        boundboxup = []
        boundboxdown = []
        # im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # for i in range(len(centcentpts)-1):
        # cv2.imshow('origin',mask*255)
        for i in range(len(centcentpts)-1):
            test_mask = mask.copy()
            cent_point = (centcentpts[i]+centcentpts[i+1])/2

            vecter = centcentpts[i+1]-centcentpts[i]
            if vecter[0]==0:
                if vecter[1]>0:
                    angle = 3.14159/2
                else:
                    angle = -3.14159/2
            else:
                if vecter[0]>0:
                    angle = math.atan(vecter[1]/vecter[0])
                else:
                    angle = math.atan(vecter[1]/vecter[0])+3.14159
            
            print(angle,'angle',vecter[1],vecter[0])
            ptss = centcentpts[i:i+2].copy() 
            color_test_mask = cv2.cvtColor(test_mask*255,cv2.COLOR_BAYER_GR2RGB)
            # cv2.circle(color_test_mask,tuple(ptss[0]),3,(0,255,0),3)
            # cv2.circle(color_test_mask,tuple(ptss[1]),3,(0,255,0),3)
            # cv2.imshow('color_test_mask',color_test_mask)
            # cv2.waitKey(0)
            res_mask,M,ptss = rotate_image(test_mask,angle,cent_point,ptss)
            
            ptss = np.maximum(0,ptss)
            # print(ptss)
            # if ptss[0][0]>ptss[1][0]:
            #     print('xxxxxxxxxxxxxxxxxxxxx================')
            #     cent_point = (ptss[0]+ptss[1])/2
            #     res_mask,M1,ptss = rotate_image(res_mask,3.14159,cent_point,ptss)
            #     M = np.dot(M1,M)
            # print(ptss)
            color_res_mask = cv2.cvtColor(res_mask*255,cv2.COLOR_BAYER_GR2RGB)
            # cv2.circle(color_res_mask,tuple(ptss[0]),3,(0,255,0),3)
            # cv2.circle(color_res_mask,tuple(ptss[1]),3,(0,255,0),3)
            # cv2.imshow('res_image',color_res_mask)
            # cv2.waitKey(0)
            sub_mask = res_mask[:,ptss[0][0]:ptss[1][0]]
            if sub_mask.shape[0] ==0 or sub_mask.shape[1]==0:
                continue
            # ## =======
            # while(True):
            #     tempim, contour, temphierarchy = cv2.findContours(sub_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #     if len(contour)<=1:
            #         break
            #     sub_mask = cv2.erode(sub_mask, kernel, iterations=1)
            
            ## =======
            tempim, contour, temphierarchy = cv2.findContours(sub_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.imshow('sub_mask',sub_mask*255)
            print(len(contour))
            # cv2.waitKey(0)
            if len(contour)==0:
                print('xxxxxxx')
                continue
            boxs = None
            for j in range(len(contour)):
                rect = cv2.minAreaRect(contour[j])
                tempboxs = cv2.boxPoints(rect)
                tempboxs = np.int0(tempboxs)
                tempboxs[:,0]+=ptss[0][0]
                # print(tempboxs)
                contour_image = np.zeros_like(subimage)
                # cv2.drawContours(contour_image,[tempboxs],0,(0,0,255),2)
                # cv2.circle(contour_image,(int(cent_point[0]),int(cent_point[1])),1,(0,255,0),2)
                # cv2.imshow('contour_image',contour_image)
                # cv2.waitKey(0)
                isin = point_in_polygon(np.array([cent_point]),tempboxs)
                # print(isin[0][0])
                if isin[0][0]:
                    boxs = tempboxs
                    break
            # print(boxs)
            if type(boxs) == type(None):
                continue
            
            print(boxs,'===============================')
            sort_boxs = np.zeros((4,2))
            for j in range(4):
                temp_vecter = boxs[j]-cent_point
                if temp_vecter[0]<0 and temp_vecter[1]<0:
                    sort_boxs[0]=boxs[j].copy()
                elif temp_vecter[0]>0 and temp_vecter[1]<0:
                    sort_boxs[1]=boxs[j].copy()
                elif temp_vecter[0]>0 and temp_vecter[1]>0:
                    sort_boxs[2]=boxs[j].copy()
                else:
                    sort_boxs[3]=boxs[j].copy()
            boxs = np.array(sort_boxs,dtype=np.int32)
            # for i in range(4):
            #     cv2.circle(color_res_mask,tuple(boxs[i]),3,(0,255,0),3)
            #     cv2.imshow('color_res_mask',color_res_mask)
            #     cv2.waitKey(0)

            boxs = np.hstack((boxs[:, :], np.ones((4, 1))))
            # print(np.dot(np.linalg.inv(M),boxs.T))
            
            boxs = (np.dot(np.linalg.inv(M),boxs.T)[0:2].T).astype(np.int32)
            print(boxs,'===============================')
            boundboxup.append(boxs[0])
            boundboxdown.append(boxs[3])
            if i == len(centcentpts)-2:
                boundboxup.append(boxs[1])
                boundboxdown.append(boxs[2])

            if DEBUG:
                cv2.drawContours(subimage,[boxs],0,(0,0,255),2)
                # for i in range(len(boxs)-1):
                #     cv2.line(sub_mask,tuple(boxs[i]),tuple(boxs[i+1]),255,3)
                cv2.imshow('sub_mask',sub_mask*255)
                cv2.imshow('subimage',subimage)
                # cv2.imshow('mask',mask*255)
                # cv2.imshow('res_image',res_mask*255)
                cv2.waitKey(0)

        boundboxup = np.array(boundboxup)
        boundboxdown.reverse()
        boundboxdown = np.array(boundboxdown)

        # print(boundboxup,boundboxdown)
        if len(boundboxup)>0:
            boundboxup[:,0]+=proposal[0]
            boundboxup[:,1]+=proposal[1]
            boundboxdown[:,0]+=proposal[0]
            boundboxdown[:,1]+=proposal[1]
            respts = np.concatenate((boundboxup,boundboxdown),0)
            ansbox.append(respts)

        if DEBUG:
            for i in range(len(boundboxup)):
                cv2.circle(image,tuple(boundboxup[i]),3,(0,255,0),3)
                # cv2.circle(image,tuple(boundboxdown[i]),3,(0,255,0),3)
            cv2.imshow('subimage',subimage)
            cv2.imshow('image',image)
            cv2.waitKey(0)
            # for i in range(len(centcentpts)):

    return np.array(ansbox)


if __name__ == '__main__':
    path = './image'
    maskpath = './mask'
    bboxpath = './bbox'
    savepath = './disp'
    for imagename in os.listdir(path):
        image = cv2.imread(os.path.join(path,imagename))
        print(imagename)
        if not imagename == '1072.jpg':
            continue
        pred_mask = np.load(maskpath+'/'+str(imagename.split('.')[0])+"_mask.npy")
        bbox = np.load(bboxpath+'/'+str(imagename.split('.')[0])+"_bbox.npy")
        # print(pred_mask,bbox)
        cv2.imshow('image',image)
        for idx,box in enumerate(bbox):
            w,h = int(box[2])-int(box[0]),int(box[3])-int(box[1])
            resize_mask = cv2.resize(pred_mask[idx],(w,h),interpolation=cv2.INTER_NEAREST)
            # resize_mask = pred_mask[idx].copy()
            erode_num = 0
            kernel = np.ones((3,3), np.uint8)
            resize_mask = np.array(resize_mask>=0.3,dtype=np.uint8) ### 
            color_resize_mask = np.zeros((resize_mask.shape[0],resize_mask.shape[1],3),dtype=np.uint8)
            X,Y = np.where(resize_mask)
            points = np.concatenate((X.reshape(-1,1),Y.reshape(-1,1)),1)
            for point in points:
                cv2.circle(color_resize_mask,(int(point[0]),int(point[1])),1,(0,0,255),1)
            print(points)
            kmeans = KMeans(n_clusters=1)  
            kmeans.fit(points)
            C = kmeans.cluster_centers_[0]
            cv2.circle(color_resize_mask,(int(C[0]),int(C[1])),1,(0,255,0),3)
            cv2.imshow('resize_mask',np.array(color_resize_mask,dtype=np.uint8))
            cv2.waitKey(0)
        # proposal_box,proposal_mask = connect(image,pred_mask,bbox)
        # for mask in proposal_mask:
        #     mask = np.array(mask,dtype=np.uint8)
        #     cv2.imshow('proposal_mask',mask*255)
        #     cv2.waitKey(0)


        # cv2.waitKey(0)
        # anspts = generate_proposal(image,pred_mask,bbox)
        # for poly in anspts:
        #     write_poly = poly[:,::-1].reshape(-1).tolist()
        #     print(write_poly)
        #     cv2.polylines(image,[poly],True,(0,0,255),3)
        # cv2.imshow('image',image)
        # cv2.imwrite(os.path.join(savepath,imagename),image)
        # cv2.waitKey(0)