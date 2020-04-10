import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model.recognition_model.DAN.Feature_Extractor import Feature_Extractor
from model.recognition_model.DAN.CAM import CAM
from model.recognition_model.DAN.DTD import DTD

class DAN(nn.Module):
    def __init__(self,opt):
        nn.Module.__init__(self)
        self.opt = opt

        from alphabet.alphabet import Alphabet
        self.n_class = len(Alphabet(opt.ADDRESS.ALPHABET))

        self.fe = Feature_Extractor(strides=[(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)],
        compress_layer=False,
        input_shape=[1, 32, 128])

        scales = self.fe.Iwantshapes()

        self.cam = CAM(scales=scales,
        maxT=25,
        depth=8,
        num_channels=64)

        self.dtd = DTD(nclass=self.n_class,
        nchannel=512,
        dropout=0.3,)

    def forward(self, input, text_length, text, text_rev, test=False):
        # data = sample_batched['image']
        # label = sample_batched['label']
        # target = encdec.encode(label)
        # Train_or_Eval(model, 'Train')
        # data = data.cuda()
        # label_flatten, length = flatten_label(target)
        # target, label_flatten = target.cuda(), label_flatten.cuda()
        # # net forward
        # features = model[0](data)
        features = self.fe(input)
        segmentation_features = self.cam(features)
        output = self.dtd(features[-1], segmentation_features, text, text_length)



        return {
            'result' : output
        }


