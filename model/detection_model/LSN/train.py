from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import config018 as config
import _init_paths

from lib.model.networkOptimier import networkOptimier
def main():
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

if __name__ == '__main__':
    main()
