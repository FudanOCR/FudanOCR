'''
MORAN模块定义了MORAN模型的组成部分，由MORN与ASRN模块拼接而成
'''
import sys
sys.path.append('/home/cjy/FudanOCR/model/recognition_model/MORAN_V2')


import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN
import cv2
import numpy as np

class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
    	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):

        '''
        初始化MORAN模型，由MORN和ASRN两部分构成

        :param int nc 图片通道数
        :param int nclass 字符表中的字符数量
        :param int nh 图片的高
        :param int targetH 经过MORN调整后图片的目标高度
        :param int targetW 经过MORN调整后图片的目标宽度
        :param bool bidirDeccoder 是否使用双向LSTM
        :param str inputDataType 数据类型
        :param int maxBatch Batch的最大数量
        :param bool CUDA 是否使用CUDA
        '''


        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)

    def forward(self, x, length, text, text_rev, test=False, debug=False):
        if debug:
            x_rectified, demo = self.MORN(x, test, debug=debug)

            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            '''
            nx0=x_rectified[0,:,:,:].permute(1, 2, 0).cpu().data.numpy()
            print('nx0', np.shape(nx0))
            nx=np.tile(nx0,(1,3))
            nx=255*np.reshape(nx,[32,100,3])
            print('nx',np.shape(nx))
            cv2.imwrite('./nx.jpg',nx)
            '''
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds


class newMORAN(nn.Module):

    def __init__(self,opt):

        from alphabet.alphabet import Alphabet

        # self.nclass = len(alphabet)
        self.nclass = len(Alphabet(opt.ADDRESS.ALPHABET))
        self.nh = opt.nh
        self.targetH = opt.targetH
        self.targetW = opt.targetW
        self.BidirDecoder = opt.BidirDecoder
        self.inputDataType = opt.inputDataType
        self.maxBatch = opt.maxBatch
        self.CUDA = opt.CUDA
        self.nc = opt.IMAGE.IMG_CHANNEL




    # def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False,
    # 	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):

        '''
        初始化MORAN模型，由MORN和ASRN两部分构成

        :param int nc 图片通道数
        :param int nclass 字符表中的字符数量
        :param int nh 图片的高
        :param int targetH 经过MORN调整后图片的目标高度
        :param int targetW 经过MORN调整后图片的目标宽度
        :param bool bidirDeccoder 是否使用双向LSTM
        :param str inputDataType 数据类型
        :param int maxBatch Batch的最大数量
        :param bool CUDA 是否使用CUDA
        '''


        super(newMORAN, self).__init__()
        self.MORN = MORN(self.nc, self.targetH, self.targetW, self.inputDataType, self.maxBatch, self.CUDA)
        self.ASRN = ASRN(self.targetH, self.nc, self.nclass, self.nh, self.BidirDecoder, self.CUDA)

    def forward(self, x, length, text, text_rev, test=False, debug=False):
        if debug:
            x_rectified, demo = self.MORN(x, test, debug=debug)

            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            '''
            nx0=x_rectified[0,:,:,:].permute(1, 2, 0).cpu().data.numpy()
            print('nx0', np.shape(nx0))
            nx=np.tile(nx0,(1,3))
            nx=255*np.reshape(nx,[32,100,3])
            print('nx',np.shape(nx))
            cv2.imwrite('./nx.jpg',nx)
            '''
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds