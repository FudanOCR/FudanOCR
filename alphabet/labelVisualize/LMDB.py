import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np

class lmdbDataset(Dataset):
    '''
    Dataset是torch的数据集基类，lmdbDataset继承该类，通过读取lmdb文件夹，获得图片-标签对
    '''

    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):
        '''
        :param str root LMDB文件的路径
        :param torchvision.transforms transform 对数据集需要做何种变换
        :param bool reverse 是否需要使用双向LSTM,构造逆标签
        :param str alphabet 字符表
        '''

        assert alphabet != None

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.alphabet = alphabet
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # print("在tools.dataset里,",index)
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            label = ''.join(label[i] if label[i].lower() in self.alphabet else ''
                            for i in range(len(label)))
            # print("在tools.dataset里,label为",label)
            if len(label) <= 0:
                return self[index + 1]
            if self.reverse:
                label_rev = label[-1::-1]
                label_rev += '$'
            label += '$'
            label = label.lower()

            if self.transform is not None:
                img = self.transform(img)

        if self.reverse:
            return (img, label, label_rev)
        else:
            return (img, label)

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

class randomSequentialSampler(sampler.Sampler):
    '''
    随机序列采样
    每次采样时随机选定一个起始点，然后根据该起始点采样一个连续序列。有可能采样到连续样本
    '''
    def __init__(self, data_source, batch_size):
        '''

        :param torch.utils.data.Dataset data_source 数据集
        :param int batch_size 批大小
        '''
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        '''
        example:
        sampler = randomSequentialSampler()
        len(sampler)的值为样本量
        '''
        return self.num_samples

    def __iter__(self):
        '''
        构建一个迭代器

        :return iter 返回一个索引列表的迭代器，每个迭代的位置表明该时刻访问的对象下标
        '''
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)
