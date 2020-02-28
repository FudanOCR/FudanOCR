from torch.utils.data import sampler
import torch
import random

# sampler=LMDB.randomSequentialSampler(dataset, cfg.MODEL.BATCH_SIZE)

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

def getSampler(opt,dataset):

    if opt.DATASETS.SAMPLER == 'Random_Sequential':
        return randomSequentialSampler(dataset, opt.MODEL.BATCH_SIZE)

    elif opt.DATASETS.SAMPLER == 'Random':
        import torch.utils.data.RandomSampler as RandomSampler
        '''不放回采样'''
        return RandomSampler(dataset,replacement=False)

    elif opt.DATASETS.SAMPLER == 'Sequential':
        import torch.utils.data.SequentialSampler as SequentialSampler
        return SequentialSampler(dataset)

    else:
        return None





