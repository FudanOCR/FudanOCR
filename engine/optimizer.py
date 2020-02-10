'''
损失函数模块
通过传入的参数返回一个已经填充好参数的loss函数
'''

import torch
import torch.optim as optim

# lossDict = {
#
#     'MSELoss' : self
#
#
# }

# # 优化器，可以为其写一个类
if opt.adam:
    optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
elif opt.sgd:
    optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)



class Optimizer(object):

    def __init__(self):
        '''
        通过传入的配置文件寻找相应的损失函数，并返回
        '''
        pass

    @staticmethod
    def getOptimizer(model,opt=''):
        def Adam(opt):
            '''
            交叉熵损失函数
            '''
            return optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        def Adadelta(opt):
            '''
            平方差损失函数
            '''
            return optim.Adadelta(MORAN.parameters(), lr=opt.lr)

        def test(opt=''):
            print("hello")

        # 获取loss函数的名称

