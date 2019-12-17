# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def train_LSN(config_file):

    import sys
    sys.path.append('./detection_model/LSN')

    from config.config import config, init_config
    from yacs.config import CfgNode as CN


    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)

    from lib.model.networkOptimier import networkOptimier
    print(opt.net)
    print(config.trainDatasetroot)
    init_config(config, config_file)
    print(config.net)
    print(config.trainDatasetroot)
    trainDatasetroot = config.trainDatasetroot
    testDatasetroot = config.testDatasetroot
    modelHome = config.modelHome
    outputModelHome = config.outputModelHome
    outputLogHome = config.outputLogHome
    net = config.net
    data = config.data
    modelPath=config.modelPath
    epoch = config.epoch
    maxEpoch = config.maxEpoch
    baselr=config.baselr
    steps=config.steps
    decayRate=config.decayRate
    valDuration=config.valDuration
    snapshot = config.snapshot
    GPUID = 1
    no = networkOptimier(trainDatasetroot, testDatasetroot, modelHome, outputModelHome, outputLogHome, net=net,data=data,GPUID=GPUID,resize_type=config.resize_type)
    no.trainval(modelPath=modelPath, epoch=epoch, maxEpoch=maxEpoch, baselr=baselr , steps=steps, decayRate=decayRate, valDuration=valDuration, snapshot=snapshot)

