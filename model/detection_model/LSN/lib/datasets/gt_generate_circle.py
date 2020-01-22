import os
import cv2
import numpy as np
import math
import pickle
from scipy import interpolate  
from lib.datasets.generate_anchors import generate_anchors
from lib.model.utils.cython_bbox import bbox_overlaps
from lib.datasets.bbox_transform import bbox_transform,bbox_transform_inv
import numpy.random as npr
import time
import warnings

warnings.filterwarnings('ignore')

DEBUG = False

def xmult(p1, p2, p0):
    return (p1[:,0]-p0[0])*(p2[1]-p0[1])-(p2[0]-p0[0])*(p1[:,1]-p0[1])

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def disptoline(p, l1, l2):
    return np.array(abs(xmult(p, l1, l2))/max(1e-5,distance(l1, l2)))

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

def changegt(ptss):
    dis1 = max(dist(ptss[0:7]),dist(ptss[7:14]))
    dis2 = dist(ptss[6:8])
    newptss = []
    if dis1<dis2:
        x = np.array([ptss[0,0],ptss[1,0]])
        y = np.array([ptss[0,1],ptss[1,1]])
        
        if abs(ptss[0][0] - ptss[1][0]) > abs(ptss[0][1]-ptss[0][1]):
            f=interpolate.interp1d(x,y,kind="linear")
            xnew = np.linspace(x[0],x[1],7)
            ynew = f(xnew)
        else:
            f=interpolate.interp1d(y,x,kind="linear")
            ynew = np.linspace(y[0],y[1],7)
            xnew = f(ynew)
        for i in range(7):
            newptss.append([xnew[i],ynew[i]])
        x = np.array([ptss[6,0],ptss[7,0]])
        y = np.array([ptss[6,1],ptss[7,1]])
        if abs(ptss[0][0] - ptss[1][0]) > abs(ptss[0][1]-ptss[0][1]):
            f=interpolate.interp1d(x,y,kind="linear")
            xnew = np.linspace(x[0],x[1],7)
            ynew = f(xnew)
        else:
            f=interpolate.interp1d(y,x,kind="linear")
            ynew = np.linspace(y[0],y[1],7)
            xnew = f(ynew)
        for i in range(7):
            newptss.append([xnew[i],ynew[i]])
        #print('change')
        return np.array(newptss)
    else:
        return ptss
    

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

def FindIntersectCandidate(candidate,gt,threshold):
    r1 = candidate[:,2].reshape(-1,1)
    r2 = gt[:,2].reshape(1,-1)
    s1 = np.zeros((r1.shape[0],r2.shape[1]))
    s1[...] = r1**2
    s2 = np.zeros((r1.shape[0],r2.shape[1]))
    s2[...] = r2**2
    x1 = candidate[:,1].reshape((-1,1))
    x2 = gt[:,1].reshape((1,-1))
    y1 = candidate[:,0].reshape((-1,1))
    y2 = gt[:,0].reshape((1,-1))
    dx = x1 - x2
    dy = y1 - y2
    d = np.sqrt(dx**2 + dy **2)
    I = np.zeros_like(d)
    idx = d>=r1+r2
    idx_ = np.where(idx)
    I[idx_] = 0
    idx2 = r1 - r2>=d
    idx2_ = np.where(idx2)
    I[idx2_] = np.pi*s2[idx2_]
    idx3 = r2-r1>=d
    idx3_ = np.where(idx3)
    I[idx3_] = np.pi*s1[idx3_]
    idx4 = ~(idx|idx2|idx3)
    angle1 = np.arccos((s1+d**2-s2)/(2*d*r1))
    angle2 = np.arccos((s2+d**2-s1)/(2*d*r2))
    tmp = s1*angle1+s2*angle2-np.sin(angle1)*r1*d
    I[idx4] = tmp[idx4]
    IoU = I/(s1*np.pi+s2*np.pi-I)
    ioumax = np.max(IoU,axis=1)

    idx5 = ioumax>=threshold
    idx6 = np.argmax(IoU,axis=1) 
    max_iou_idx = np.argmax(IoU,axis=0)
    max_iou_idx_pos = np.where(IoU[max_iou_idx,np.arange(0,len(max_iou_idx))]>=0.2)[0]
    _max_iou_idx = max_iou_idx[max_iou_idx_pos]
    reg = np.zeros_like(candidate)
    pos = np.zeros((candidate.shape[0]))

    tr = np.log(r2/r1)
    tx = (x2 - x1)/r1
    ty = (y2 - y1)/r1
    idx5_ = np.where(idx5)
    pos[_max_iou_idx] = 1
    pos[idx5_] = 1
    reg[idx5_,2] = tr[(idx5_[0],idx6[idx5_])]
    reg[idx5_,1] = tx[(idx5_[0],idx6[idx5_])]
    reg[idx5_,0] = ty[(idx5_[0],idx6[idx5_])]

    reg[_max_iou_idx,2] = tr[_max_iou_idx,max_iou_idx_pos]
    reg[_max_iou_idx,1] = tx[_max_iou_idx,max_iou_idx_pos]
    reg[_max_iou_idx,0] = ty[_max_iou_idx,max_iou_idx_pos]
    return pos,reg

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4
    # print(bbox_transform(ex_rois, gt_rois[:, :4]))
    return bbox_transform(ex_rois, gt_rois).astype(np.float32, copy=False)

def generate_gt(image,ptss,bbx,strides):
    vis = np.zeros((image.shape[0]*image.shape[1]))
    R = np.zeros((image.shape[0]*image.shape[1]))
    gtx = np.array([])
    gty = np.array([])
    gtr = np.array([])
    for idx,pts in enumerate(ptss):
        box = bbx[idx]
        maxr = max(abs(box[0]-box[2]),abs(box[1]-box[3]))
        linecenter = []
        pts = changegt(pts)
        pts = np.array(pts,dtype = np.int32)
        begineh = dist(np.array([pts[0],pts[13]]))
        endh = dist(np.array([pts[6],pts[7]]))
        polyheight = []
        for i in range(7):
            linecenter.append((pts[i]+pts[13-i])/2)
            polyheight.append(dist(np.vstack((pts[i],pts[13-i]))))
        linecenter = np.array(linecenter)
        # print('linecenter',linecenter)
        vectorbegine = linecenter[0]-linecenter[1]
        vectorend = linecenter[6]-linecenter[5]
        if not vectorbegine[0]+vectorbegine[1] == 0:
            linecenter[0] = (1-begineh*0.3/dist(linecenter[:2]))*vectorbegine+linecenter[1]
        if not vectorend[0]+vectorend[1] == 0:
            linecenter[6] = (1-endh*0.3/dist(linecenter[5:]))*vectorend+linecenter[5]
        # cv2.polylines(showimage,[np.array(pts,dtype=np.int32)],True,(0,255,0),1)
        # cv2.polylines(showimage2,[np.array(pts,dtype=np.int32)],True,(0,255,0),1)
        linecenter = np.array(linecenter,dtype=np.int32)
        for i in range(6):
            # print('_-------------------------------------------------------_\n')
            lencenter = dist(np.vstack((linecenter[i],linecenter[i+1])))
            num = lencenter/(min(polyheight[i],polyheight[i+1])/8+1)
            if abs(linecenter[i+1][0] - linecenter[i][0]) >= abs(linecenter[i+1][1]-linecenter[i][1]):
                xnew = np.linspace(linecenter[i][0],linecenter[i+1][0],int(num))
                # print(linecenter[i][0],linecenter[i+1][0])
                f=interpolate.interp1d([linecenter[i][0],linecenter[i+1][0]],[linecenter[i][1],linecenter[i+1][1]],kind="linear",fill_value=-1)
                ynew = f(xnew)
            else:
                ynew = np.linspace(linecenter[i][1],linecenter[i+1][1],int(num))
                # print(linecenter[i][1],linecenter[i+1][1])
                f=interpolate.interp1d([linecenter[i][1],linecenter[i+1][1]],[linecenter[i][0],linecenter[i+1][0]],kind="linear",fill_value=-1)
                xnew = f(ynew)
            
            xnew = np.array(xnew,dtype = np.int32)
            ynew = np.array(ynew,dtype = np.int32)
            # print('xxxxxx\n')
            # print('xnew',xnew)
            pnew = np.hstack((xnew[:,np.newaxis],ynew[:,np.newaxis]))
            rnew = np.minimum(disptoline(pnew,pts[i],pts[i+1]),disptoline(pnew,pts[13-i],pts[12-i]))
            subnew = ynew*(image.shape[1])+xnew
            vis[subnew] = rnew<maxr
            R[subnew] = rnew
    labelptssub = np.where(vis==1)[0]
    R = np.array(R,dtype=np.int32)
    gty = labelptssub//image.shape[1]
    gtx = labelptssub%image.shape[1]
    gtr = R[labelptssub]
    gt = np.hstack((gtx[:,np.newaxis],gty[:,np.newaxis],gtr[:,np.newaxis]))
    # for circle in gt:
    #     circle= np.array(circle,dtype=np.int32)
    #     cv2.circle(showimage3,(circle[0],circle[1]),circle[2],(0,255,0),3)
    gt = np.array(gt,dtype=np.float32)
    circle_labels = {}
    circle_regres = {}
    anchor_labels = {}
    anchor_regres = {}
    anchor_positive_weights = {}
    anchor_negative_weights = {}
    all_circles = {}
    all_anchors = {}
    for stride in strides:
        # print(stride)
        # circle_labels[stride] = np.zeros((image.shape[0]//stride,image.shape[1]//stride,2))
        _circles = np.array([[0,0,stride],[0,0,stride*np.sqrt(2)]])
        _num_circles = _circles.shape[0]
        # Enumerate all shifts
        shift_x = np.arange(0, (image.shape[1]//stride)) * stride
        shift_y = np.arange(0, (image.shape[0]//stride)) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift_r = np.zeros_like(shift_x)
        shifts = np.vstack(np.vstack((shift_x.ravel(), shift_y.ravel(),shift_r.ravel())).transpose())
        A = _num_circles
        K = shifts.shape[0]
        circles = _circles.reshape((1, A, 3)) + \
                shifts.reshape((1, K, 3)).transpose((1, 0, 2))
        circles = circles.reshape((K * A, 3))
        is_in_polygon = np.zeros((len(circles)))
        for pts in ptss:
            inpolygon = point_in_polygon(circles,pts)
            is_in_polygon = is_in_polygon + inpolygon.flatten()
        circles_in_polygon_idx = np.where(is_in_polygon>0)[0]
        # print(len(anchors),len(gt),len(anchors_in_polygon_idx))
        pos = np.zeros((circles.shape[0]))
        # pos.fill(-1)
        reg = np.zeros_like(circles)
        # reg.fill(-1)
        if len(circles_in_polygon_idx)>0:
            subpos,subreg = FindIntersectCandidate(circles[circles_in_polygon_idx],gt,0.4)
            pos[circles_in_polygon_idx] = subpos
            reg[circles_in_polygon_idx] = subreg
        circle_labels[stride] = pos
        circle_regres[stride] = reg
        all_circles[stride] = circles
        # pos_idx = np.where(pos==1)[0]
        # for i in pos_idx:
        #     px = anchors[i][0]
        #     py = anchors[i][1]
        #     pr = anchors[i][2]
        #     dx = reg[i][0]
        #     dy = reg[i][1]
        #     dr = reg[i][2]
        #     cv2.circle(showimage,(int(px),int(py)),int(pr),(255,0,0),3)
        #     pxnew = int(pr*dx+px)
        #     pynew = int(pr*dy+py)
        #     prnew = int(np.exp(dr)*pr)
        #     cv2.circle(showimage2,(pxnew,pynew),prnew,(0,0,255),3)
        # _anchors = generate_anchors(base_size = stride,scales=np.array([1,np.sqrt(2),np.sqrt(3)]),ratios=np.array([1.0/8,1.0/4,1.0/2,1]))
        # _num_anchors = _anchors.shape[0]
        # # Enumerate all shifts
        # shift_x = np.arange(0, (image.shape[1]//stride)) * stride
        # shift_y = np.arange(0, (image.shape[0]//stride)) * stride
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # shifts = np.vstack(np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                         shift_x.ravel(), shift_y.ravel())).transpose())
        # A = _num_anchors
        # K = shifts.shape[0]
        # anchors = _anchors.reshape((1, A, 4)) + \
        #         shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        # stride_anchors = anchors.reshape((K * A, 4))
        # inds_inside = np.where(
        #     (stride_anchors[:, 0] >= 0) &
        #     (stride_anchors[:, 1] >= 0) &
        #     (stride_anchors[:, 2] <= image.shape[1]) &  # width
        #     (stride_anchors[:, 3] <= image.shape[0])  # height
        # )[0]
        # all_rois = stride_anchors[inds_inside,:]
        # circle_labels_stride_anchors = np.empty((len(stride_anchors),), dtype=np.float32)
        # circle_labels_stride_anchors.fill(-1)
        # gt_boxes = np.array(bbx)
        # labels = np.empty((len(all_rois),), dtype=np.float32)
        # labels.fill(-1)
        # overlaps = bbox_overlaps(
        #     np.ascontiguousarray(all_rois, dtype=np.float),
        #     np.ascontiguousarray(gt_boxes, dtype=np.float))
        # argmax_overlaps = overlaps.argmax(axis=1)
        # max_overlaps = overlaps[np.arange(len(all_rois)), argmax_overlaps]
        # gt_argmax_overlaps = overlaps.argmax(axis=0)
        # gt_max_overlaps = overlaps[gt_argmax_overlaps,
        #                             np.arange(overlaps.shape[1])]
        # gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        
        # # fg label: for each gt, anchor with highest overlap
        # labels[gt_argmax_overlaps] = 1

        # # fg label: above threshold IOU
        # labels[max_overlaps >= 0.5] = 1 # POSITIVE_OVERLAP

        # # assign bg labels first so that positive labels can clobber them
        # labels[max_overlaps < 0.1 ] = 0 # 

        # # subsample positive labels if we have too many
        # num_fg = 200
        # fg_inds = np.where(labels == 1)[0]
        # if len(fg_inds) > num_fg:
        #     disable_inds = npr.choice(
        #         fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        #     labels[disable_inds] = -1

        # num_bg = 400  - np.sum(labels == 1)
        # bg_inds = np.where(labels == 0)[0]
        # if len(bg_inds) > num_bg:
        #     disable_inds = npr.choice(
        #         bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        #     labels[disable_inds] = -1

        # bbox_targets = np.zeros((len(all_rois), 4), dtype=np.float32)
        # # print('origin:',bbox_targets)
        # bbox_targets[:,:] = _compute_targets(all_rois, gt_boxes[argmax_overlaps, :])
        # # print(bbox_targets)

        # # draw
        # # bbox_reg = bbox_transform_inv(all_rois, bbox_targets)
        # # anchors_pos_idx = np.where(labels==1)[0]
        # # for anchor_idx in anchors_pos_idx:
        # #     anchors = np.array(bbox_reg[anchor_idx],dtype=np.int32)
        # #     cv2.rectangle(anchorimage,(anchors[0],anchors[1]),(anchors[2],anchors[3]),(0,255,0),3)

        # num_examples = np.sum(labels >= 0)
        # # print(num_examples)
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples

        # circle_labels_stride_anchors[inds_inside] = labels
        # regres_stride_anchors = np.zeros((len(stride_anchors), 4), dtype=np.float32)
        # regres_stride_anchors[inds_inside] = bbox_targets
        # anchor_labels[stride] = circle_labels_stride_anchors
        # anchor_regres[stride] = regres_stride_anchors
        # anchor_positive_weights[stride] = positive_weights
        # anchor_negative_weights[stride] = negative_weights
        # all_anchors[stride] = stride_anchors
    # cv2.imshow('anchorimage',anchorimage)
    # cv2.imshow('image',showimage)
    # cv2.imshow('showimage2',showimage2)
    # cv2.imshow('showimage3',showimage3)
    # cv2.imwrite('./disp/rec_'+imagename,anchorimage)
    # cv2.imwrite('./disp/anchor_'+imagename,showimage)
    # cv2.imwrite('./disp/proposal_'+imagename,showimage2)
    # cv2.imwrite('./disp/gt_'+imagename,showimage3)
    # cv2.waitKey(0)
    return circle_labels,circle_regres,all_circles#,anchor_labels,anchor_regres,all_anchors,anchor_positive_weights,anchor_negative_weights
    

if __name__ == '__main__':
    datasetroot = '/home/wangxiaocong/fudan_ocr_system/datasets/ICDAR15/Text_Localization/ch4_training_images'
    imagelist = os.listdir(datasetroot+'/text_image')
    samplelist = []
    for imagename in imagelist:
        samplelist.append([datasetroot+'/text_image/'+imagename,datasetroot+'/text_label_curve/'+imagename.split('.')[0]+'.txt'])
    maxR = 0
    minR = 10000000
    proposalR = 2**(np.linspace(6,15,10))
    strides = [8,16,32,64,128]
    for sample in samplelist:
        print(sample[0])
        # if not sample[0] == './data/ctw1500/train/text_image/0006.jpg':
        #     continue
        imagename = sample[0].split('/')[-1]
        image,ptss,bbx = loadData(sample)
        image,ptss,bbx = resize(image,ptss,bbx,800,1432)
        generate_gt(image,ptss,bbx,strides)
    print(maxR,minR)