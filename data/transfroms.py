import random
import torch
import torchvision
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import lmdb
import six
import sys
import numpy as np
import cv2
from PIL import Image


def build_transforms(cfg, is_train=True):
    transform = None

    if cfg.BASE.MODEL == 'MORAN':
        transform = resizeNormalsize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == "TextSnake":
        transform = NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)

    elif cfg.BASE.MODEL == 'maskrcnn':
        transform = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )

    return transform




#处理类

class Resize(object):
    # 对图片进行尺度调整
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_jittering = False
        if isinstance(min_size, tuple):
            self.min_size_random_group = min_size
            self.scale_jittering = True
            self.group_size = len(min_size)

    # modified from torchvision to add support for max size
    def get_size(self, image_size):

        w, h = image_size
        size = self.min_size
        if self.scale_jittering:
            size = self.min_size_random_group[np.random.randint(self.group_size)]

        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        # print('size:', (oh, ow))
        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(target.size)  # sth wrong with image.size
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class resizeNormalize(object):
    '''
    将图片进行放缩，并标准化
    '''

    def __init__(self, size, interpolation=Image.BILINEAR):
        '''

        :param tuple size 需要将原图变换至目标尺寸
        :param interpolation 插值方法
        '''
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        '''
        传入一张图片，将图片放缩后并进行标准化，像素放缩到[-1,1]的位置

        :param Image img 图片
        '''
        print('value of size is', self.size)
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NewAugmentation(object):

    def __init__(self, size, mean, std, maxlen, minlen):
        self.size = size
        self.mean = mean
        self.std = std
        self.maxlen = maxlen
        self.minlen = minlen
        self.augmentation = Compose([
            # RandomResize(scale_list=[0.5, 1.0, 2.0], minlen=minlen),
            RandomMirror(),
            Rotate(),
            RandomCrop(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


