# -*- coding:utf-8 -*-
import os
from PIL import Image, ImageFilter
import cv2
import random
import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as Transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", '.tif', '.tiff'])


def load_img(filepath, type='RGB'):
    img = Image.open(filepath).convert(type)
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDataset(Data.Dataset):
    def __init__(self, image_dir, crop_size=512, scale_factor=4,
                 random_scale=True, rotate=True, fliplr=True, fliptb=True):
        super(TrainDataset, self).__init__()

        self.image_dir = image_dir
        self.image_filenames = []
        self.image_filenames.extend(os.path.join(image_dir, x)
                                    for x in sorted(os.listdir(image_dir))
                                    if is_image_file(x))
        
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.random_scale = random_scale
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb

        h_w_scale = 1
        self.crop_size_h = calculate_valid_crop_size(self.crop_size // h_w_scale, self.scale_factor)
        self.crop_size_w = self.crop_size_h * h_w_scale

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # determine valid HR image size with scale factor
        hr_img_w = self.crop_size_w
        hr_img_h = self.crop_size_h

        # determine LR image size
        lr_img_w_2x = hr_img_w // (self.scale_factor // 2)
        lr_img_h_2x = hr_img_h // (self.scale_factor // 2)
        lr_img_w_4x = hr_img_w // self.scale_factor
        lr_img_h_4x = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size_w:
                ratio = self.crop_size_W / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size_h:
                ratio = self.crop_size_h / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            transform = Transforms.Resize(
                (scale_h, scale_w), interpolation=Image.ANTIALIAS)
            img = transform(img)

        # random crop on image
        transform = Transforms.RandomCrop((self.crop_size_h, self.crop_size_w))
        img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(0, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = Transforms.RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        hr_img = Transforms.CenterCrop((hr_img_h, hr_img_w))(img)
        lr2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.ANTIALIAS)(hr_img)
        lr4x_img = Transforms.Resize((lr_img_h_4x, lr_img_w_4x), interpolation=Image.ANTIALIAS)(hr_img)
        bc2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.BICUBIC)(lr4x_img)
        bc4x_img = Transforms.Resize((hr_img_h, hr_img_w), interpolation=Image.BICUBIC)(lr4x_img)

        # Tensor Transform
        img_transform = Transforms.ToTensor()

        hr_img = img_transform(hr_img)
        lr2x_img = img_transform(lr2x_img)
        lr4x_img = img_transform(lr4x_img)
        bc2x_img = img_transform(bc2x_img)
        bc4x_img = img_transform(bc4x_img)

        # print(hr_img.size())
        # print(lr2x_img.size())
        # print(lr4x_img.size())
        # print(bc2x_img.size())
        # print(bc4x_img.size())

        return hr_img, lr2x_img, lr4x_img, bc2x_img, bc4x_img

    def __len__(self):
        return len(self.image_filenames)


class DevDataset(Data.Dataset):
    def __init__(self, image_dir):
        super(DevDataset, self).__init__()

        self.image_dir = image_dir
        self.image_filenames = []
        self.image_filenames.extend(os.path.join(image_dir, x)
                                    for x in sorted(os.listdir(image_dir))
                                    if is_image_file(x))

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])
        width = img.size[0]
        height = img.size[1]

        # determine LR image size
        lr_img_w_4x = width // 4
        lr_img_h_4x = height // 4
        lr_img_w_2x = lr_img_w_4x * 2
        lr_img_h_2x = lr_img_h_4x * 2
        hr_img_w = lr_img_w_4x * 4
        hr_img_h = lr_img_h_4x * 4
        
        hr_img = Transforms.Resize((hr_img_h, hr_img_w))(img)
        lr2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.ANTIALIAS)(hr_img)
        lr4x_img = Transforms.Resize((lr_img_h_4x, lr_img_w_4x), interpolation=Image.ANTIALIAS)(hr_img)
        bc2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.BICUBIC)(lr4x_img)
        bc4x_img = Transforms.Resize((hr_img_h, hr_img_w), interpolation=Image.BICUBIC)(lr4x_img)

        # Tensor Transform
        img_transform = Transforms.ToTensor()

        hr_img = img_transform(hr_img)
        lr2x_img = img_transform(lr2x_img)
        lr4x_img = img_transform(lr4x_img)
        bc2x_img = img_transform(bc2x_img)
        bc4x_img = img_transform(bc4x_img)

        return hr_img, lr2x_img, lr4x_img, bc2x_img, bc4x_img

    def __len__(self):
        return len(self.image_filenames)



class TestDataset(Data.Dataset):
    def __init__(self, image_dir):
        super(TestDataset, self).__init__()

        self.image_dir = image_dir
        self.image_filenames = []
        self.image_filenames.extend(os.path.join(image_dir, x)
                                    for x in sorted(os.listdir(image_dir))
                                    if is_image_file(x))

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])
        width = img.size[0]
        height = img.size[1]

        # determine LR image size
        lr_img_w_4x = width // 4
        lr_img_h_4x = height // 4
        lr_img_w_2x = lr_img_w_4x * 2
        lr_img_h_2x = lr_img_h_4x * 2
        hr_img_w = lr_img_w_4x * 4
        hr_img_h = lr_img_h_4x * 4
        
        hr_img = Transforms.Resize((hr_img_h, hr_img_w))(img)
        lr2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.ANTIALIAS)(hr_img)
        lr4x_img = Transforms.Resize((lr_img_h_4x, lr_img_w_4x), interpolation=Image.ANTIALIAS)(hr_img)
        bc2x_img = Transforms.Resize((lr_img_h_2x, lr_img_w_2x), interpolation=Image.BICUBIC)(lr4x_img)
        bc4x_img = Transforms.Resize((hr_img_h, hr_img_w), interpolation=Image.BICUBIC)(lr4x_img)

        # Tensor Transform
        img_transform = Transforms.ToTensor()

        hr_img = img_transform(hr_img)
        lr2x_img = img_transform(lr2x_img)
        lr4x_img = img_transform(lr4x_img)
        bc2x_img = img_transform(bc2x_img)
        bc4x_img = img_transform(bc4x_img)

        return hr_img, lr2x_img, lr4x_img, bc2x_img, bc4x_img

    def __len__(self):
        return len(self.image_filenames)
