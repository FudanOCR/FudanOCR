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


class Env(object):

    def __init__(self):

        # self.readCommand()
        self.seedInit()
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
        print("Parameters: ", arg.config_file)

        '''
        TODO 将配置文件的路径传给config模块
        '''

    def createFolder(self, rootList, removeOrigin=False):
        '''
        新建文件夹操作
        removeOrigin用于判断是否删除原有文件
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





