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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import PIL


def getTransforms(cfg,split="train" ):
    transform = None

    if cfg.BASE.MODEL == 'MORAN':
        transform = resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'GRCNN':
        transform = resizeNormalizeAndPadding(cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)
        # transform = None

    elif cfg.BASE.MODEL == "TextSnake":
        transform = NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)

    elif cfg.BASE.MODEL == 'CRNN':
        transform = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'RARE':
        transform = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'SAR':
        transform = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'DAN':
        transform = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'AON':
        if split == 'val':
            transform = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))
        elif split == 'train':
            '''Rotate to a random angle'''
            transform = resizeNormalizeAndGrayAndRandom90Rotate((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))


    elif cfg.BASE.MODEL == 'maskrcnn':
        transform = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )

    return transform

def getTranforms_by_assignment(cfg ):
    transforms = None
    if cfg.DATASETS.TRANSFORM == 'Normalize' :
        transforms = Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
        )
    elif cfg.DATASETS.TRANSFORM == 'resizeNormalizeAndGray':
        transforms = resizeNormalizeAndGray((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.DATASETS.TRANSFORM == 'resizeNormalizeAndPadding':
        transforms = resizeNormalizeAndPadding(cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)

    elif cfg.DATASETS.TRANSFORM == 'resizeNormalize':
        transforms = resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.DATASETS.TRANSFORM == 'NewAugmentation':
        transforms = NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)

    elif cfg.DATASETS.TRANSFORM == 'Compose':
        #need modify to adapt a list of strings
        transforms = Compose(
            [
                '''
                Resize(min_size, max_size),
                ToTensor(),
                MixUp(mix_ratio=0.1),
                Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255),
                '''
            ]
        )

    return  transforms

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
        # print('value of size is', self.size)
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class resizeNormalizeAndGrayAndRandomRotate(object):
    '''
    将图片进行放缩，并标准化，转化为灰度图
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
        # print('value of size is', self.size)
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
            # img = img.resize(self.size, self.interpolation)
            # img = img.convert('L')
        img = cv2.resize(img, self.size)
        '''Rotate'''
        # print(img.shape)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        import random
        angel = random.randint(0, 360)

        M = cv2.getRotationMatrix2D(center, angel, 0.71)
        img = cv2.warpAffine(img, M, (w, h))
        # print("旋转之后",img.shape)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class resizeNormalizeAndGrayAndRandom90Rotate(object):
    '''
    将图片进行放缩，并标准化，转化为灰度图
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
        # print('value of size is', self.size)
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
            # img = img.resize(self.size, self.interpolation)
            # img = img.convert('L')
        img = cv2.resize(img, self.size)
        '''Rotate'''
        # print(img.shape)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        import random
        angel = random.randint(0, 4)

        M = cv2.getRotationMatrix2D(center, angel*90,1.0)
        img = cv2.warpAffine(img, M, (w, h))
        # print("旋转之后",img.shape)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class resizeNormalizeAndGray(object):
    '''
    将图片进行放缩，并标准化，转化为灰度图
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
        # print('value of size is', self.size)
        if isinstance(img, PIL.Image.Image):
            img = img.resize(self.size, self.interpolation)
            img = img.convert('L')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.size)
        # img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class resizeNormalizeAndPadding(object):

    def __init__(self, maxW, imgH, interpolation=Image.BILINEAR):
        self.imgH = imgH
        self.maxW = maxW
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        try:
            ratio = img.shape[1] / img.shape[0]
        except:
            ratio = img.size[1] / img.size[0]
        # print(img.size)
        imgW = int(self.imgH * ratio)
        # print("resize weight:", img.shape, imgW, self.imgH)
        # img = img.resize((imgW, self.imgH), self.interpolation)

        # print("插入调试信息")
        # print(type(img))
        # # print(imgW)
        # print("结束调试信息")

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        try:
            # img = cv2.imread('/home/cjy/test.jpg')
            # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (imgW, self.imgH))
        except:
            print("读取图片发生错误，使用替代图片")
            img = cv2.imread('/home/cjy/test.jpg')
            ratio = img.shape[1] / img.shape[0]
            # print(img.size)
            imgW = int(self.imgH * ratio)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (imgW, self.imgH))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        padding = (0, 0, self.maxW - imgW, 0)
        img = ImageOps.expand(img, border=padding, fill='black')
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

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

