'''
损失函数模块
通过传入的参数返回一个已经填充好参数的loss函数
'''

import torch
import torch.optim as optim


# # 优化器，可以为其写一个类
# if opt.adam:
#     optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# elif opt.adadelta:
#     optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
# elif opt.sgd:
#     optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
# else:
#     optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)


def getOptimizer(model,opt=''):
    def Adam(opt):
        '''
        交叉熵损失函数
        '''
        return optim.Adam(model.parameters(), lr=opt.MODEL.LR, betas=(0.5, 0.999))

    def Adadelta(opt):
        '''
        平方差损失函数
        '''
        return optim.Adadelta(model.parameters(), lr=opt.MODEL.LR)


        # 获取loss函数的名称

    '''
    Loss字典，根据参数文件的字符串选择对应的函数
    '''
    optimizerDict = {
        'Adadelta': Adadelta,
        'Adam': Adam,
    }

    return optimizerDict[opt.MODEL.OPTIMIZER](opt)

