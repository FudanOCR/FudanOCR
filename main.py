from yacs.config import CfgNode as CN
from train.moran_v2 import train_moran_v2
from train.grcnn import train_grcnn
from train.moran_v2_xuxixi import train_moran_v2_xuxixi
from train.east import train_east
# from train import AdvancedEAST
# from train.
# from train import AdvancedEAST
import argparse
import re

import sys
sys.path.append('./recognition_model/')
sys.path.append('./detection_model/')
print("当前系统环境变量为：",sys.path)

# 在这个位置扩充函数
function_dict = {

    'MORAN_V2': train_moran_v2,
    'MORAN_V2_xuxixi' : train_moran_v2_xuxixi,
    'GRCNN': train_grcnn,
    'EAST': train_east,
    #  'AdvancedEAST': AdvancedEAST
   # 'AdvancedEAST': AdvancedEAST,
   # 'TextSnake':, 
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
