# -*- coding:utf-8 -*-
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def norm(imgs, vgg=True):
    # normalize for pre-trained vgg model
    if vgg:
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    # normalize [-1, 1]
    else:
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])   

    size = list(imgs.shape)
    res = imgs.clone()
    if len(size) == 4:
        for i in range(size[0]):
            res[i] = transform(res[i])
    else:
        res = transform(imgs)

    return res

def denorm(imgs, vgg=True):
    size = list(imgs.shape)
    res = imgs.clone()
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                         std=[4.367, 4.464, 4.444])
        if len(size) == 4:
            for i in range(size[0]):
                res[i] = transform(imgs[i])
        else:
            res = transform(res)
    else:
        if len(size) == 4:
            for i in range(size[0]):
                res[i] = ((res[i] + 1) / 2).clamp(0, 1)
        else:
            res = (res + 1) / 2
            res = res.clamp(0, 1)
    
    return res