import torch
import torch.nn as nn
def getVGG16(opt):

    '''cnn'''
    nc = opt.IMAGE.IMG_CHANNEL
    '''
    nm: chanel number
    ks: kernel size
    ps: padding size
    ss: stride size
    '''
    nm = [64, 128, 256, 256, 512, 512, 512]
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]

    cnn = nn.Sequential()

    def convRelu(i, batchNormalization=False, leakyRelu=False):
        nIn = nc if i == 0 else nm[i - 1]
        nOut = nm[i]
        cnn.add_module('conv{0}'.format(i),
                       nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        if batchNormalization:
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

        if leakyRelu:
            cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    # 32 * 100
    convRelu(0, False)
    # 32 * 100
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    # 16 * 50
    convRelu(1, False)
    # 16 * 50
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
    # 8 * 25
    convRelu(2, True)
    convRelu(3, False)
    # 8 * 25
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
    # # 4 * 27
    convRelu(4, True)
    convRelu(5, False)
    # 4 * 27
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
    # 2 * 29
    convRelu(6, True)
    # 1 * ?
    # 也就是说，当图片的高为32时，经过卷积层之后，输出的特征图维度的高将变为1


    return cnn