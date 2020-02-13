'''
环境模块
包含对实验环境的操作，例如初始化随机数种子
'''

import os
import shutil
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import argparse
from config.config import get_cfg_defaults
from yacs.config import CfgNode as CN

class Env(object):

    def __init__(self):
        '''
        进行一系列初始化之后，将命令行参数给的配置文件读出来，交予类变量self
        '''

        self.opt = self.readCommand()
        self.seedInit()
        self.setVisibleGpu()
        self.setCudnnBenchmark()

    def seedInit(self):
        '''
        随机数种子初始化
        '''
        manualSeed = random.randint(1, 10000)

        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    def setCudnnBenchmark(self):
        '''
        设置cudnn.benchmark
        '''
        cudnn.benchmark = True

    def readCommand(self):
        '''
        读取命令行参数
        将读取到的配置文件交给config读取模块
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', required=True, help='path to config file')
        arg = parser.parse_args()
        # print("Parameters CONFIG_FILE: ", arg.config_file)

        # cfg = get_cfg_defaults()
        # cfg.merge_from_file(arg.config_file)
        # cfg.freeze()
        # return cfg

        opt = self.read_config_file(arg.config_file)
        return opt


    def read_config_file(self,config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    def createFolder(self, rootList, removeOrigin=False):
        '''
        新建文件夹操作
        removeOrigin用于判断是否删除原有文件
        TODO 从参数文件中解析出Address部分
        '''

        if isinstance(rootList, str):
            rootList = [rootList]

        if removeOrigin == True:
            for root in rootList:
                if os.path.exists(root):
                    shutil.rmtree(root)
                os.makedirs(root)
        else:
            for root in rootList:
                if os.path.exists(root):
                    print('Path always exists: ',root)
                    shutil.rmtree(root)
                os.makedirs(root)


    def getOpt(self):
        '''
        返回解析好的配置文件opt
        '''
        return self.opt

    def setVisibleGpu(self):
        '''
        设置可用gpu编号
        '''
        gpu_list = [str(i) for i in self.opt.BASE.GPU_ID]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_list)



