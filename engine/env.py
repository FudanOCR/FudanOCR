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
import pynvml
from config.yaml_reader import read_config_file
from model.modelDict import getModel


class Env(object):

    def __init__(self):
        '''
        进行一系列初始化之后，将命令行参数给的配置文件读出来，交予类变量self
        '''

        self.opt = self.readCommand()
        self.seedInit()
        self.setVisibleGpu()
        self.setCudnnBenchmark()
        self.checkAddressExist()
        self.model = getModel(self.opt.BASE.MODEL)

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

        opt = read_config_file(arg.config_file)

        '''这里需要对opt做进一步处理'''

        def deleteLastLine(str):
            while str[-1] == '/' or str[-1] == '\\':
                str = str[:-1]
            return str

        opt.ADDRESS.CHECKPOINTS_DIR = deleteLastLine(opt.ADDRESS.CHECKPOINTS_DIR) + '_' + opt.BASE.EXPERIMENT_NAME
        opt.ADDRESS.LOGGER_DIR = deleteLastLine(opt.ADDRESS.LOGGER_DIR) + '_' + opt.BASE.EXPERIMENT_NAME
        opt.VISUALIZE.TAG = deleteLastLine(opt.VISUALIZE.TAG) + '_' + opt.BASE.EXPERIMENT_NAME

        # print("opt.BASE.EXPERIMENT_NAME is not defined. Please update your yaml according to the example.yaml.")

        return opt


    # def read_config_file(self,config_file):
    #     # 用yaml重构配置文件
    #     f = open(config_file)
    #     opt = CN.load_cfg(f)
    #     return opt

    def getOpt(self):
        '''
        返回解析好的配置文件opt
        '''
        return self.opt

    def setVisibleGpu(self):
        '''
        设置可用gpu编号
        '''
        num_gpu = self.opt.BASE.NUM_GPUS
        gpu_list = [str(i) for i in self.opt.BASE.GPU_ID]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_list[:num_gpu])

        '''检测gpu使用情况'''
        import pynvml
        pynvml.nvmlInit()
        # 这里的1是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_list[0]))
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = meminfo.total  # 第二块显卡总的显存大小
        used = meminfo.used  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2

        ratio = used / total
        if ratio > 0.5:
            flag = True
            while flag == True:
                ans = input("More than 50% resource has been occupied on GPU{0}, are you sure to continue?(y/n)".format(str(gpu_list[0])))
                if ans == 'n':
                    exit(0)
                elif ans== 'y':
                    flag = False





    def checkAddressExist(self):
        '''
        检查路径是否存在
        路径分为两类：
        1.指定了就一定要存在：例如数据集文件夹
        2.指定了并不一定要存在，如果不存在即由程序创建：例如checkpoint
        '''

        if self.opt.FUNCTION.VAL_ONLY:
            return

        def folderExist(key,value):
            '''
            对于必须存在的路径的检查
            如果是空value，则不需要考虑
            '''
            if value == '':
                return

            if os.path.exists(value):
                pass
            else:
                assert False, "Address " + key + " : " + value + ' doesn\'t exist!'

        def createFolder(rootList, removeOrigin=False):
            '''
            对于不必要存在的路径，将由程序处理
            removeOrigin用于判断是否删除原有文件
            TODO 从参数文件中解析出Address部分
            '''

            if isinstance(rootList, str):
                '''
                不考虑空字符串
                '''
                if rootList == '':
                    return
                rootList = [rootList]

            if removeOrigin == True:
                for root in rootList:
                    if os.path.exists(root):
                        shutil.rmtree(root)
                    os.makedirs(root)
            else:
                for root in rootList:
                    if os.path.exists(root):
                        print('Path always exists: ', root)
                    else:
                        print('Make folder: ' , root)
                        os.makedirs(root)

        model_type = self.opt.BASE.TYPE
        if model_type == 'D':
            assert self.opt.ADDRESS.DET_RESULT_DIR != ''
            assert self.opt.ADDRESS.GT_JSON_DIR != ''
        if model_type == 'R':
            folderExist('opt.ADDRESS.ALPHABET', self.opt.ADDRESS.ALPHABET)

        folderExist('opt.ADDRESS.TRAIN_DATA_DIR', self.opt.ADDRESS.TRAIN_DATA_DIR)
        folderExist('opt.ADDRESS.TRAIN_GT_DIR', self.opt.ADDRESS.TRAIN_GT_DIR)
        folderExist('opt.ADDRESS.TEST_DATA_DIR', self.opt.ADDRESS.TEST_DATA_DIR)
        folderExist('opt.ADDRESS.TEST_GT_DIR', self.opt.ADDRESS.TEST_GT_DIR)
        folderExist('opt.ADDRESS.VAL_DATA_DIR', self.opt.ADDRESS.VAL_DATA_DIR)
        folderExist('opt.ADDRESS.VAL_GT_DIR', self.opt.ADDRESS.VAL_GT_DIR)




        createFolder(self.opt.ADDRESS.CHECKPOINTS_DIR,removeOrigin=True)
        # createFolder(self.opt.ADDRESS.CACHE_DIR)
        '''保证每次训练时使用的文件夹都是新的'''
        createFolder(self.opt.ADDRESS.LOGGER_DIR,removeOrigin=True)


