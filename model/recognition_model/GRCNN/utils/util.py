#!/usr/bin/python
# encoding: utf-8

'''
util.py 定义了一些工具函数，包含数据集的生成等
'''


import torch
import torch.nn as nn
import collections
import random
import numpy as np
import cv2
from faker import Faker
#from CNumber import cnumber

import datetime


class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + ' '  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text, depth=0):
        """Support batch or single str."""
        if isinstance(text, str):
            # text = text.lower()
            for char in text:
                # Fix the bug
                if self.alphabet.find(char) == -1:
                    print(char)
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)

        if depth:
            return text, len(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list).replace(' ','')
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class Generator(object):
    """An abstract class for text generator.
    一个抽象类，用于数据集的生成，等待被实现
    """
    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class CnumberGenerator(Generator):
   def __init__(self):
       self.cnum = cnumber()

   def __len__(self):
       return 128000

   def __getitem__(self, index):
       num = random.randint(100, 9999999)
       if random.randint(0, 1):
           num = num / 100.0
       return self.cnum.cwchange(num)

class TextGenerator(Generator):
    """Invoice message txt generator
    args:
        texts: File path which contains 
    """
    def __init__(self, texts, len_thr):
        super(TextGenerator, self).__init__()
        self.len_thr = len_thr
        with open(texts) as f:
            self.texts = f.readlines()

    def __getitem__(self, index):
        text_len = len(self.texts[index])
        if text_len > self.len_thr:
            text_len = self.len_thr
        return self.texts[index].strip()[0:text_len]

    def __len__(self):
        return len(self.texts)

    def __len_thr__(self):
        return self.len_thr

class PasswordGenerator(Generator):
    def __init__(self):
        self.fake = Faker()
        self.fake.random.seed(4323)

    def __getitem__(self, index):
        return self.fake.password(length=10, special_chars=True, digits=True, upper_case=True, lower_case=True)
        
    def __len__(self):
        return 320000


class HyperTextGenerator(Generator):
    def __init__(self, texts):
        self.invoice_gen = TextGenerator(texts)
        #self.passwd_gen = PasswordGenerator()
        self.cnum_gen = CnumberGenerator()
    
    def __getitem__(self, index):
        rnd = random.randint(0, 1)
        if rnd:
            cur = index % self.invoice_gen.__len__()
            return self.invoice_gen.__getitem__(cur)
        else:
            return self.cnum_gen.__getitem__(index)

    def __len__(self):
        return self.invoice_gen.__len__() + self.cnum_gen.__len__()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0], v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

def getValidateCheckout(id17):
    '''获得校验码算法'''
    weight=[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2] #十七位数字本体码权重   
    validate=['1','0','X','9','8','7','6','5','4','3','2'] #mod11,对应校验码字符值   

    sum=0
    mode=0
    for i in range(0,len(id17)):
        sum = sum + int(id17[i])*weight[i]
    mode=sum%11
    return validate[mode]

def getRandomIdNumber(sex = 1):
    '''产生随机可用身份证号，sex = 1表示男性，sex = 0表示女性'''
    #地址码产生
    from addr import addr #地址码
    addrInfo = random.randint(0,len(addr) - 1)#随机选择一个值
    addrId = addr[addrInfo][0]
    addrName = addr[addrInfo][1]
    idNumber = str(addrId)
    #出生日期码
    start, end = "1960-01-01","2000-12-30" #生日起止日期
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    birthDays = datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(random.randint(0,days)), "%Y%m%d")
    idNumber = idNumber + str(birthDays)
    #顺序码
    for i in range(2):#产生前面的随机值
        n = random.randint(0,9)# 最后一个值可以包括
        idNumber = idNumber + str(n)
    # 性别数字码
    sexId = random.randrange(sex,10,step = 2) #性别码
    idNumber = idNumber + str(sexId)
    # 校验码
    checkOut = getValidateCheckout(idNumber)
    idNumber = idNumber + str(checkOut)
    return idNumber

def getRandomAddress(addr_path):
    """Generate random address."""
    with open(addr_path) as f:
        metalist = f.readlines()

    province = list()
    city = list()
    district = list()
    biz_area = list()
    for meta in metalist:
        addr, t = meta.strip().split(",")
        addr = addr.strip()
        t = t.strip()
        if t == "province":
            province.append(addr)
        elif t == "city":
            city.append(addr)
        elif t == "district":
            district.append(addr)
        else:
            biz_area.append(addr)

    road = ['弄', '栋', '号']

    part_addr = list()
    part_addr.append(random.choice(province))
    part_addr.append(random.choice(city))
    part_addr.append(random.choice(district))
    part_addr.append(random.choice(biz_area))
    part_addr.append(str(random.randint(1, 9999)))
    part_addr.append(random.choice(road))

    return ''.join(part_addr)



def r(val):
    """
    Generate random number.
    """
    return int(np.random.random() * val)

def random_scale(x,y):
    ''' 对x随机scale,生成x-y之间的一个数'''
    gray_out = r(y+1-x) + x
    return gray_out

def text_Gengray(bg_gray, line):
    gray_flag = np.random.randint(2)
    if bg_gray < line:
        text_gray = random_scale(bg_gray + line, 255)
    elif bg_gray > (255 - line):
        text_gray = random_scale(0, bg_gray - line)
    else:
        text_gray = gray_flag*random_scale(0, bg_gray - line) + (1 - gray_flag)*random_scale(bg_gray+line, 255)
    return text_gray

def Addblur(img, val):
    blur_kernel = random_scale(2,val)
    img = cv2.blur(img, (blur_kernel,blur_kernel))
    return img

def motionBlur(img, val):
    blur_kernel0 = random_scale(2,val)
    blur_kernel1 = random_scale(2,val)
    anchor = (random_scale(0,blur_kernel0-1),random_scale(0,blur_kernel1-1))
    img = cv2.blur(img,(blur_kernel0,blur_kernel1),anchor=anchor)
    return img

def AffineTransform(img, val):
    rows,cols = img.shape[:2]
    ang = random.randint(0,10)
    theta = ang * np.pi / 180
    M = np.eye(2,3) # affine matrix
    M[0,1] = np.tan(theta)
    #M = M + np.random.random((2,3)) * val
    res = cv2.warpAffine(img, M, (cols,rows))
    return res

def AddNoiseSingleChannel(single):
    diff = (255 - single.max()) / 3
    noise = np.random.normal(0, 1+r(6), single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def randomNoise(img, max_noise_num=10):
    max_brightness = np.max(img)
    noise_num = random.randint(0,max_noise_num)

    pos = np.where(img > 20)

    if len(pos[0]) > 0:
        for i in range(noise_num):
            id = random.randint(0, len(pos[0])-1)
            y, x = pos[0][id], pos[1][id]
            cv2.circle(img, (x,y), 2, 0, -1)
    return img

