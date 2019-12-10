import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN
import cv2
import numpy as np

class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
    	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
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
