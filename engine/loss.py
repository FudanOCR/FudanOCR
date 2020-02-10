'''
损失函数模块
通过传入的参数返回一个已经填充好参数的loss函数
'''

import torch

# lossDict = {
#
#     'MSELoss' : self
#
#
# }




class Loss(object):

    def __init__(self):
        '''
        通过传入的配置文件寻找相应的损失函数，并返回
        '''
        pass


    @staticmethod
    def getLoss(opt=''):



        def CrossEntropyLoss(opt):
            '''
            交叉熵损失函数
            '''
            return torch.nn.CrossEntropyLoss()

        def MSELoss(opt):
            '''
            平方差损失函数
            '''
            return torch.nn.MSELoss()

        def test(opt=''):
            print("hello")

        # 获取loss函数的名称

