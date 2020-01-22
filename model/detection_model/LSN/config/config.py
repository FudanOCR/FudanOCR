import sys
import torch
import os
from easydict import EasyDict
from yacs.config import CfgNode as CN

config = EasyDict()

config.net = 'resnet50'
config.data = 'icdar' #synthtext
config.filename = 'config'

config.datasetroot = '/home/shf/fudan_ocr_system/datasets/ICDAR15/Text_Localization'
config.trainDatasetroot = '/home/shf/fudan_ocr_system/datasets/ICDAR15/Text_Localization/test'
config.testDatasetroot = '/home/shf/fudan_ocr_system/datasets/ICDAR15/Text_Localization/train'
config.modelHome = '/home/shf/fudan_ocr_system/LSN/pretrainmodel'
config.outputModelHome = '/home/shf/fudan_ocr_system/LSN/lib/model/data/2019AAAI/output/'
config.outputLogHome = '/home/shf/fudan_ocr_system/LSN/lib/model/data/2019AAAI/output/'

config.bbox = '/home/shf/fudan_ocr_system/LSN/lib/model/utils/bbox.pyx'
config.modelPath='/home/shf/fudan_ocr_system/LSN/pretrainmodel/resnet50.pth'
config.epoch = 0
config.maxEpoch = 1000

config.baselr=0.0001
config.steps=[1000]
config.decayRate=0.1
config.valDuration= 5
config.snapshot = 5
config.resize_type = 'normal'

def init_config(config, config_file):
    f = open(config_file)
    opt = CN.load_cfg(f)

    config.net = opt.net
    config.data = opt.data
    config.filename = opt.filename

    config.datasetroot = opt.datasetroot
    config.trainDatasetroot = opt.trainDatasetroot
    config.testDatasetroot = opt.testDatasetroot
    config.modelHome = opt.modelHome
    config.outputModelHome = os.path.join(opt.outputModelHome, opt.net, 'model/', opt.filename)
    config.outputLogHome = os.path.join(opt.outputLogHome, opt.net, 'model/', opt.filename)

    config.bbox = opt.bbox
    config.modelPath = opt.modelPath
    config.epoch = opt.epoch
    config.maxEpoch = opt.maxEpoch

    config.baselr = opt.baselr
    config.steps = opt.steps
    config.decayRate = opt.decayRate
    config.valDuration = opt.valDuration
    config.snapshot = opt.snapshot
    config.resize_type = opt.resize_type


