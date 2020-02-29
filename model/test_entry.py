# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
sys.path.append('./recognition_model/')
sys.path.append('./detection_model/')
sys.path.append("./super_resolution_model/")
sys.path.append("./maskrcnn_benchmark_architecture/")
print("当前系统环境变量为：",sys.path)


from test.moran_v2 import test_moran_v2
from test.AdvancedEAST import test_AdvancedEAST
from test.grcnn import test_grcnn
# from train.fasterrcnn import train_fasterrcnn
from test.east import test_east
from test.TextSnake import test_TextSnake
from test.PSENet import test_psenet
from test.DocumentSRModel import test_documentsrmodel
from test.HARN import test_HARN
from test.PixelLink import test_PixelLink
from test.maskscoring_rcnn import test_maskscoring_rcnn
from test.LSN import test_LSN

from yacs.config import CfgNode as CN
import argparse
import re

# 在这个位置扩充函数
function_dict = {

    'MORAN_V2': test_moran_v2,
    'AdvancedEAST': test_AdvancedEAST,
    'GRCNN': test_grcnn,
    'EAST': test_east,
    # 'fasterrcnn': train_fasterrcnn,
    'TextSnake': test_TextSnake,
    'PSENet' : test_psenet,
    'DocumentSRModel' : test_documentsrmodel,
    'HARN': test_HARN,
    'PixelLink': test_PixelLink,
    'maskscoring_rcnn': test_maskscoring_rcnn,
    'LSN': LSN,
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
