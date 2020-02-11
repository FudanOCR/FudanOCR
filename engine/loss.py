'''
损失函数模块
通过传入的参数返回一个已经填充好参数的loss函数
'''

import torch



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

    # 获取loss函数的名称

    '''
    Loss字典，根据参数文件的字符串选择对应的函数
    '''
    lossDict = {
        'MSELoss': MSELoss,
        'CrossEntropyLoss': CrossEntropyLoss,
    }

    return lossDict[opt.MODEL.LOSS](opt)

