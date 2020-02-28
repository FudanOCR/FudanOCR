import torch
import random
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


class alignCollate(object):

    def __init__(self, opt, keep_ratio=True, min_ratio=1):
        """
        args:
            imgH: can be divided by 32
            maxW: the maximum width of the collection
            keep_ratio:
            min_ratio:
        """
        self.imgH = opt.IMAGE.IMG_H
        self.imgW = opt.IMAGE.IMG_W
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    # 解耦
    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                try:
                    w, h = image.size
                except:
                    w, h = image.shape[1], image.shape[0]  # image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(self.min_ratio * imgH, imgW)  # assure imgW >= imgH

        '''导入外部变换'''


        transform = resizeNormalize(imgW, imgH)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

class resizeNormalize(object):

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

def getCollate(opt,dataset):

    if opt.DATASETS.COLLATE_FN == 'ALIGN_COLLATE':
        return alignCollate(opt)
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate





