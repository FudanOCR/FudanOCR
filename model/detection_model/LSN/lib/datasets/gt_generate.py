import os
import cv2
import numpy as np
import math
# from lib.datasets.generate_anchors import generate_anchors
from lib.datasets.proposal_generate import ProposalGenerate
import pickle

DEBUG = False

def dist(pts):
    ret = 0.0
    for i in range(pts.shape[0]-1):
        ret += np.linalg.norm(pts[i]-pts[i+1])
    return ret

def triarea(a,b,c):
    return 0.5*(a[0]*b[1]+b[0]*c[1]+c[0]*a[1]-a[0]*c[1]-b[0]*a[1]-c[0]*b[1])

def mycross(p1,p2,p3): # 叉积判定
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1     

def segment(p1,p2,p3,p4): #判断两线段是否相交
    if(max(p1[0],p2[0])>=min(p3[0],p4[0]) and max(p3[0],p4[0])>=min(p1[0],p2[0]) and max(p1[1],p2[1])>=min(p3[1],p4[1]) and max(p3[1],p4[1])>=min(p1[1],p2[1])): #矩形2最高端大于矩形1最低端
        if(mycross(p1,p2,p3)*mycross(p1,p2,p4)<=0 and mycross(p3,p4,p1)*mycross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
    else:
        D=0
    return D

def point_in_polygon(point,polygon):
    c = False
    i = -1
    l = len(polygon)
    j = l - 1
    while i < l-1:
        i += 1
        if ((polygon[i][0] <= point[0] and point[0] < polygon[j][0]) or (polygon[j][0] <= point[0] and point[0] < polygon[i][0])):
            if (point[1] < (polygon[j][1] - polygon[i][1]) * (point[0] - polygon[i][0]) / (polygon[j][0] - polygon[i][0]) + polygon[i][1]):
                c = not c
        j = i
    return c

def polyarea(pts):
    area = 0.0
    for i in range(len(pts)):
        begin,end = pts[i],pts[(i+1)%(len(pts))]
        area += triarea(begin,end,[0,0])
    return np.abs(area)

def within(point,b):
    for i in range(b.shape[0]):
        begin,end = b[i],b[(i+1)%b.shape[0]]
        area = triarea(begin,end,point)
        if area<0:
            return False,True#status,isillegal
        if np.abs(area)<1e-5 and i%2==0:
            return False,False
    return True,True

def rot(pts):
    ptsnew = []
    ptsnew.append(pts[6])
    begin = pts[6]
    end = pts[7]
    duration = (end - begin)/6.0
    for i in range(1,6,1):
        ptsnew.append((begin+duration*i).astype(np.int))
    ptsnew.append(pts[7])
    ptsnew.append(pts[13])
    begin = pts[13]
    end = pts[0]
    duration = (end - begin)/6.0
    for i in range(1,6,1):
        ptsnew.append((begin+duration*i).astype(np.int))
    ptsnew.append(pts[0])
    return np.array(ptsnew,dtype=np.float32)

def loadData(sample):
    image = cv2.imread(sample[0])
    canvas = image.copy()
    labelfile = open(sample[1])
    lines = labelfile.readlines()
    labelfile.close()
    bbx = []
    ptss = []
    for line in lines:
        points = line.strip().split(',')
        points = list(map(int,points))
        xmin,ymin,xmax,ymax = (points[0],points[1],points[2],points[3])
        pts = np.array(points[4:],dtype = np.float32).reshape(-1,2)
        pts[:,0]+=xmin
        pts[:,1]+=ymin
        avglong = dist(pts[0:7,:]) + dist(pts[7:14,:])
        avgshort = dist(pts[6:8,:]) + dist(np.vstack(([pts[13,:],pts[0,:]])))
        if avglong < avgshort:
            pts = rot(pts)
        if DEBUG:
            for i in range(14):
                cv2.line(canvas,(int(pts[i,0]),int(pts[i,1])),(int(pts[(i+1)%14,0]),int(pts[(i+1)%14,1])),(0,0,255),1)
        bbx.append([xmin,ymin,xmax,ymax])
        ptss.append(pts)
    ptss = np.ascontiguousarray(ptss,np.float32)
    bbx = np.ascontiguousarray(bbx,dtype=np.float32)
    if DEBUG:
        print('show')
        cv2.imshow('image',canvas)
        cv2.waitKey(30)
    return image,ptss,bbx

def resize(image,ptss,bbx,preferredShort,maxLong):
    h,w,c = image.shape
    shortSide,longSide = min(h,w),max(h,w)
    ratio = preferredShort*1.0/shortSide
    if longSide*ratio>maxLong:
        ratio = maxLong*1.0/longSide
    retimage = cv2.resize(image,None,None,ratio,ratio)
    ptss*=ratio
    bbx*=ratio
    return retimage,ptss,bbx

def rotate(image,ptss,bbx,angle):
    if angle%360 == 0:
        return image,ptss,bbx
    h,w,c = image.shape
    print(h,w)
    _cos = np.cos(angle/180.0*3.1416)
    _sin = np.sin(angle/180.0*3.1416)

    M1 = np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
    M2 = np.array([[_cos,_sin,0],[-_sin,_cos,0],[0,0,1]])

    tmp = np.dot(np.dot(M2,M1),np.array([[0,0,w,w],[0,h,h,0],[1,1,1,1]]))
    
    left = np.floor(np.min(tmp[0]))
    top = np.floor(np.min(tmp[1]))
    right = np.ceil(np.max(tmp[0]))
    bottom = np.ceil(np.max(tmp[1]))
    print(left,top,right,bottom)
    new_w = right - left + 1
    new_h = bottom - top +1

    M3 = np.array([[1,0,new_w/2],[0,1,new_h/2],[0,0,1]])
    M = np.dot(M3,np.dot(M2,M1))

    for i in range(ptss.shape[0]):
        pts = np.hstack((ptss[i,:,:],np.ones((14,1))))
        pts = np.dot(M,pts.T)[0:2].T
        bbx[i,0] = np.min(pts[:,0])
        bbx[i,1] = np.min(pts[:,1])
        bbx[i,2] = np.max(pts[:,0])
        bbx[i,3] = np.max(pts[:,1])

        ptss[i,:,:] = pts

    retimage = cv2.warpAffine(image,M[0:2],(int(new_w),int(new_h)))
    return retimage,ptss,bbx

if __name__ == '__main__':
    datasetroot = './data/ctw15/train'
    imagelist = os.listdir(datasetroot+'/text_image')
    PG = ProposalGenerate(8,2**np.linspace(0,5,11),[1])
    samplelist = []
    for imagename in imagelist:
        samplelist.append([datasetroot+'/text_image/'+imagename,datasetroot+'/text_label_curve/'+imagename.split('.')[0]+'.txt'])
    for sample in samplelist:
        print(sample[0])
        # sample
        # if sample[0]!='./data/ctw15/train/text_image/0023.jpg':
        #     continue
        image,ptss,bbx = loadData(sample)
        image,ptss,bbx = resize(image,ptss,bbx,960,2048)
        padimage = np.zeros((int(math.ceil(image.shape[0]*1.0/16)*16),int(math.ceil(image.shape[1]*1.0/16)*16),3),dtype = np.uint8)
        padimage[0:image.shape[0],0:image.shape[1],:] = image
        showimage = padimage.copy()
        anchors = PG.run(padimage.shape[0]/8,padimage.shape[1]/8,[padimage.shape[0],padimage.shape[1],1],0,ptss,image,bbx)
        cv2.imshow('showimage',showimage)
        cv2.waitKey(0)
        # for idx,pts in enumerate(ptss):
        #     # showimage = padimage.copy()
        #     pts = np.array(pts,dtype = np.int32)
        #     # print(pts)
        #     centerpts = np.zeros_like(pts)
        #     height = 0
        #     for i in range(7):
        #         center = ((pts[i][0]+pts[13-i][0])*1.0/2,(pts[i][1]+pts[13-i][1])*1.0/2)
        #         # print(center)
        #         height+=np.sqrt((pts[i][0]-pts[13-i][0])*(pts[i][0]-pts[13-i][0])+(pts[i][1]-pts[13-i][1])*(pts[i][1]-pts[13-i][1]))
        #         top = ((pts[i][0]-center[0])*1.0/2+center[0],(pts[i][1]-center[1])*1.0/2+center[1])
        #         bottom = ((pts[13-i][0]-center[0])*1.0/2+center[0],(pts[13-i][1]-center[1])*1.0/2+center[1])
        #         centerpts[i]=top
        #         centerpts[13-i]=bottom
        #     # print()
        #     height=height*1.0/7;
        #     box = bbx[idx]
        #     # print(pts)
        #     # print(centerpts)
        #     cv2.polylines(showimage,[pts],True,(0,0,255),3)
        #     cv2.polylines(showimage,[centerpts],True,(255,0,0),3)
        #     # mask = np.zeros((showimage.shape[0],showimage.shape[1]),dtype= np.uint8)
        #     # mask = cv2.fillPoly(mask,[pts],1)
        #     # polyarea = np.sum(mask)
        #     subanchors = PG.run((box[3]-box[1])/8+4,(box[2]-box[0])/8+4,(box[3]-box[1]+(box[3]-box[1])/2,box[2]-box[0]+(box[3]-box[1])/2,3),(box[3]-box[1])/2)
        #     subanchors[:,1] = subanchors[:,1]+box[0]-2
        #     subanchors[:,3] = subanchors[:,3]+box[0]-2
        #     subanchors[:,2] = subanchors[:,2]+box[1]-2
        #     subanchors[:,4] = subanchors[:,4]+box[1]-2
            
            
        #     for idx,anchor in enumerate(subanchors):
        #         LT = point_in_polygon((anchor[1],anchor[2]),pts)
        #         RT = point_in_polygon((anchor[3],anchor[2]),pts)
        #         RB = point_in_polygon((anchor[3],anchor[4]),pts)
        #         LB = point_in_polygon((anchor[1],anchor[4]),pts)
        #         top = 0
        #         bottom = 0
        #         for i in range(6):
        #             top = (top or segment((anchor[1],anchor[2]),(anchor[3],anchor[2]),pts[i],pts[i+1]))
        #             bottom = (bottom or segment((anchor[1],anchor[2]),(anchor[3],anchor[2]),pts[13-i],pts[12-i]))

        #             top = (top or segment((anchor[3],anchor[2]),(anchor[3],anchor[4]),pts[i],pts[i+1]))
        #             bottom = (bottom or segment((anchor[3],anchor[2]),(anchor[3],anchor[4]),pts[13-i],pts[12-i]))

        #             top = (top or segment((anchor[3],anchor[4]),(anchor[1],anchor[4]),pts[i],pts[i+1]))
        #             bottom = (bottom or segment((anchor[3],anchor[4]),(anchor[1],anchor[4]),pts[13-i],pts[12-i]))

        #             top = (top or segment((anchor[1],anchor[4]),(anchor[1],anchor[2]),pts[i],pts[i+1]))
        #             bottom = (bottom or segment((anchor[1],anchor[4]),(anchor[1],anchor[2]),pts[13-i],pts[12-i]))

        #         # if(not LT and not RB) or (not RT and not LB):
        #         anchorcenter = ((anchor[1]+anchor[3])/2,(anchor[2]+anchor[4])/2)
        #         anchorheight = min(anchor[3]-anchor[1],anchor[4]-anchor[2])
        #         if point_in_polygon(anchorcenter,centerpts) and top and bottom and anchorheight/height<1.8:
        #             # print(top,bottom)
        #             # subshowimage = showimage.copy()
        #             subanchors[i][0]=1
        #             # print(subanchors)
        #             cv2.rectangle(showimage,(anchor[1],anchor[2]),(anchor[3],anchor[4]),(0,255,0),3)
        #     cv2.imshow('subshowimage',showimage)
        #     cv2.imwrite(sample[0].split('/')[-1],showimage)
        #     cv2.waitKey(30)
        #     if(len(anchors)>0):
        #         anchors = np.vstack((anchors,subanchors))
        #     else:
        #         anchors = subanchors
        
                # if(not LT and not RB) or (not RT and not LB):
                    # submask = mask.copy()
                    # submask[int(anchor[2]):int(anchor[4]),int(anchor[1]):int(anchor[3])]=0
                    # subpolyarea = np.sum(submask)
                    # recarea = (anchor[3]-anchor[1])*(anchor[4]-anchor[2])
                    # # print((polyarea-subpolyarea)*1.0/recarea)
                    # if (polyarea-subpolyarea)*1.0/recarea > 0.5:
                    #     cv2.rectangle(showimage,(anchor[1],anchor[2]),(anchor[3],anchor[4]),(0,255,0),3)
                    #     cv2.imshow('image',showimage)
                    #     # cv2.imshow('submask',submask*255)
                    #     cv2.waitKey(0)
        # for anchor in anchors:
        #     print(anchor)
        # print(ptss)
        