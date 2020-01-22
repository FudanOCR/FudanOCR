# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
sys.path.append('./recognition_model/')
sys.path.append('./detection_model/')
sys.path.append("./super_resolution_model/")
sys.path.append("./maskrcnn_benchmark_architecture/")
print("当前系统环境变量为：",sys.path)


from train.moran_v2 import train_moran_v2
from train.grcnn import train_grcnn
from train.maskrcnn_architecture import train_maskrcnn_architecture
from train.east import train_east
from train.TextSnake import train_TextSnake
from train.PSENet import train_psenet
from train.AdvancedEAST import train_AEAST
from train.DocumentSRModel import train_documentsrmodel
from train.HARN import train_HARN
from train.LSN import train_LSN
from train.PixelLink import train_PixelLink
from train.maskscoring_rcnn import train_maskscoring_rcnn

from yacs.config import CfgNode as CN
import argparse
import re

# 在这个位置扩充函数
function_dict = {

    'MORAN_V2': train_moran_v2,
    'GRCNN': train_grcnn,
    'EAST': train_east,
    'maskrcnn_architecture': train_maskrcnn_architecture,
    'AdvancedEAST': train_AEAST,
    'TextSnake': train_TextSnake,
    'PSENet' : train_psenet,
    'DocumentSRModel' : train_documentsrmodel,
    'HARN': train_HARN,
    'LSN': train_LSN,
    'PixelLink': train_PixelLink,
    'maskscoring_rcnn': train_maskscoring_rcnn,
    'Your Model Name': 'Your Model Function'
}

def read_config_file(config_file):
    # 用yaml重构配置文件
    f = open(config_file)
    result = CN.load_cfg(f)
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True, help='path to config file')
opt = parser.parse_args()


if __name__ == '__main__':

    # 读取配置文件
    result = read_config_file(opt.config_file)
    # 通过 '_' 区分调用模型的名称，并调用函数
    model_name = result.model
    function = function_dict[model_name]
    # 调用函数，传入配置文件
    function(opt.config_file)
