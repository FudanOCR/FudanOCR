#!/usr/bin/python
# encoding: utf-8

import models.convnet as ConvNets
import models.recurrent as SeqNets
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist

class CRANN(nn.Module):
    def __init__(self, crann_config, n_class):
        super(CRANN, self).__init__()
        self.ngpu = crann_config['N_GPU']
        cnn_conf = crann_config['CNN']
        print('Constructing {}'.format(cnn_conf['MODEL']))
        self.cnn = ConvNets.__dict__[cnn_conf['MODEL']]()

        rnn_conf = crann_config['RNN']
        print('Constructing {}'.format(rnn_conf['MODEL']))
        self.rnn = SeqNets.__dict__[rnn_conf['MODEL']](rnn_conf, n_class)


    def forward(self, input):
        c_feat = data_parallel(self.cnn, input, self.ngpu)

        b, c, h, w = c_feat.size()
        #print("feature size, b:{0}, c:{1}, h:{2}, w:{3}".format(b, c, h, w))

        assert h == 1, "the height of the conv must be 1"

        c_feat = c_feat.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1) # [w, b, c]

        output = data_parallel(self.rnn, c_feat, self.ngpu, dim=1)
        
        return output

def data_parallel(model, input, ngpu, dim=0):
    #dist.init_process_group(init_method='file:///workspace/mnt/group/ocr-fd-group/zhangpeiyao/CRNN/zhang/sharedfile',backend="gloo",world_size=4,group_name="pytorch_test")
    if isinstance(input.data, torch.cuda.FloatTensor) and ngpu > 1:
        output = nn.parallel.data_parallel(model, input, range(ngpu), dim=dim)
        #output = nn.parallel.DistributedDataParallel(model, input, range(ngpu), dim=dim)
    else:
        output = model(input.cuda())#####################
    return output
