import os
import bisect
import copy
import logging

import torch.utils.data
from alphabet.alphabet import Alphabet
from data import LMDB
import data.grcnndataset as grcnndataset
# from model.detection_model.TextSnake_pytorch.dataset import total_text
# from data  importICDAR
# from data import CTW1500
from model.detection_model.AdvancedEAST.utils.data_utils import custom_dset, collate_fn
from model.detection_model.TextSnake_pytorch.util.augmentation import EvalTransform, NewAugmentation
import torch


def get_dataset(cfg, name, data_dir, anno_dir, split, alphabet):
    '''
    :param cfg:
    :param data_dir:
    :param anno_dir:
    :param split: has 'train' ,'val', 'test'
    :return: dataset
    '''

    if cfg.BASE.TYPE == 'R':
        '''
        如果是识别模型，则数据集读取方式划分为lmdb, custom
        lmdb: 数据集中的img和label被打包成一个lmdb文件夹，用户只需将lmdb文件夹路径赋予cfg.ADDRESS.TRAIN(VAL)_DATA_DIR即可
        
              例如以下lmdb文件路径，将cfg.ADDRESS.TRAIN_DATA_DIR赋值为/train_lmdb（请自行换成绝对路径）
                /train_lmdb
                    /data.mdb
                    /lock.mdb
                    
        custom: cfg.ADDRESS.TRAIN(VAL)_DATA_DIR为图片文件夹地址，cfg.ADDRESS.TRAIN(VAL)_GT_DIR为一个文本文件，记载图片地址与标签的一一映射
        
            例如以下路径，将cfg.ADDRESS.TRAIN(VAL)_DATA_DIR赋值为/img（请自行换成绝对路径）
            /img
                /1.png
                ...
                /99.png
            
            标签文件如下所示，cfg.ADDRESS.TRAIN(VAL)_GT_DIR赋值为/img_label.txt（请自行换成绝对路径）
            /img_label.txt
            标签文件的内容如下，无需带上前缀/img
            1.png hello
            ...
            99.png FudanOCR
        '''

        if 'lmdb' == name.lower():
            dataset = LMDB.lmdbDataset(root=data_dir,
                                       transform=LMDB.resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)),
                                       reverse=cfg.BidirDecoder, alphabet=alphabet.str)
            assert dataset
            return dataset

        elif 'custom' == name.lower():
            dataset = grcnndataset.CustomDataset(data_dir, anno_dir, transform=None)
            # dataset = CustomDataset(data_dir, anno_dir, transform=LMDB.resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)))

            assert dataset
            return dataset

    if 'Lmdb' in name:
        dataset = LMDB.lmdbDataset(root=data_dir,
                                   transform=LMDB.resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)),
                                   reverse=cfg.BidirDecoder, alphabet=alphabet.str)
        assert dataset
        return dataset

    elif 'custom_dset' in name:
        dataset = custom_dset(split=split)
        assert dataset
        return dataset

    if 'CTW1500' in name:
        dataset = CTW1500.CTW1500Loader(data_dir, anno_dir)
        return dataset

    # elif 'totol_text' in name:

    elif 'ICDAR2013Dataset' in name:
        dataset = icdar_seriers.ICDAR2013Dataset(data_dir, anno_dir)
        assert dataset
        return dataset

    elif 'ICDAR2015TRAIN' in name:
        dataset = icdar_seriers.ICDAR2015TRAIN(data_dir, anno_dir)
        assert dataset
        return dataset
    '''
    elif 'total_text' in name:
        if split == 'train':
            trainset = TotalText(
                data_root=data_dir,
                ignore_list=os.path.join(data_dir, 'ignore_list.txt'),
                is_training=True,
                transform=NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)
            )
            return  trainset
        if split == 'val':
            trainset = TotalText(
                data_root=data_dir,
                ignore_list=os.path.join(data_dir, 'ignore_list.txt'),
                is_training=True,
                transform=NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)
            )
            return  trainset
        if split == 'test':
            testset = TotalText(
                data_root=data_dir,
                ignore_list=os.path.join(data_dir, 'ignore_list.txt'),
                is_training=False,
                transform=EvalTransform(size=1280, mean=cfg.means, std=cfg.stds)
                # transform=BaseTransform(size=1280, mean=cfg.means, std=cfg.stds)
            )
            return  testset

    '''
    raise RuntimeError("Dataset not available: {}".format(name))


def get_dataloader(cfg, name, dataset, split):

    if cfg.BASE.TYPE == 'R':

        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=cfg.MODEL.BATCH_SIZE,
                                                   shuffle=shuffle,
                                                   sampler=None,
                                                   num_workers=cfg.BASE.WORKERS,
                                                 collate_fn=grcnndataset.alignCollate(
                                                     imgH=cfg.IMAGE.IMG_H,
                                                     imgW=cfg.IMAGE.IMG_W)
                                                 )
        assert dataloader
        return dataloader


    if 'Imdb' in name:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.MODEL.BATCH_SIZE,
            shuffle=False, sampler=LMDB.randomSequentialSampler(dataset, cfg.MODEL.BATCH_SIZE),
            num_workers=int(cfg.BASE.WORKERS))
        assert dataloader
        return dataloader

    elif 'custom_dset' in name:
        if split == 'train':
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.MODEL.BATCH_SIZE, shuffle=True,
                                                     collate_fn=collate_fn,
                                                     num_workers=int(cfg.BASE.WORKERS), drop_last=False)
        elif split == 'val':
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                                     num_workers=int(cfg.BASE.WORKERS))

        assert dataloader
        return dataloader

    elif 'ICDAR2013Dataset' in name:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=int(cfg.BASE.WORKERS),
            batch_sampler=batch_sampler,
        )
        assert dataloader
        return dataloader

    elif 'ICDAR2015TRAIN' in name:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=int(cfg.BASE.WORKERS),
            batch_sampler=batch_sampler,
        )
        assert dataloader
        return dataloader

    elif 'CTW1500' in name:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.MODEL.BATCH_SIZE,
            shuffle=True,
            num_workers=3,
            drop_last=True,
            pin_memory=True)
        assert dataloader
        return dataloader

    '''
       elif 'total_text' in name:
           if split == 'train':
               train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.MOEDL.BATCH_SIZE, shuffle=True,
                                              num_workers=int(cfg.BASE.WORKERS))
               return train_loader
           #total_text doesn't need val_data
           if split == 'val':
               train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.MOEDL.BATCH_SIZE, shuffle=True,
                                              num_workers=int(cfg.BASE.WORKERS))
               return train_loader
           if split == 'test':
               test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.int(cfg.BASE.WORKERS))
               return  test_loader
       '''

    raise RuntimeError("Dataset not available: {}".format(name))


def build_dataloader(cfg, is_train=True):
    type_name = cfg.DATASETS.TYPE
    model_type = cfg.BASE.TYPE
    # 考虑不需区分dataset是否是检测或识别,只需填入需要的地址
    if is_train == True:
        '''Use a letter to encode the type of the model'''
        if model_type == 'R':
            alphabet = Alphabet(cfg.ADDRESS.ALPHABET)
        else:
            alphabet = Alphabet()

        train_data_dir = cfg.ADDRESS.TRAIN_DATA_DIR
        train_anno_dir = cfg.ADDRESS.TRAIN_GT_DIR
        val_data_dir = cfg.ADDRESS.VAL_DATA_DIR
        val_anno_dir = cfg.ADDRESS.VAL_GT_DIR

        train_set = get_dataset(cfg, type_name, train_data_dir, train_anno_dir, split='train', alphabet=alphabet)
        val_set = get_dataset(cfg, type_name, val_data_dir, val_anno_dir, split='val', alphabet=alphabet)

        train_dataloader = get_dataloader(cfg, type_name, dataset=train_set, split='train')
        val_dataloader = get_dataloader(cfg, type_name, dataset=val_set, split='val')

        # remind the relation of batch_size and num of gpus
        images_per_batch = cfg.MODEL.BATCH_SIZE
        num_gpus = int(cfg.BASE.NUM_GPUS)
        assert (
                images_per_batch % num_gpus == 0
        ), "IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus

        if images_per_gpu > 5:
            logger = logging.getLogger(__name__)
            # logger.warning(
            #     "每GPU图片数量过高时可能遇到内存溢出，"
            #     "若发生该情况请调整BATCH_SIZE,并调整学习率等其他可能影响效果的因素"
            # )

        return train_dataloader, val_dataloader

    else:
        if model_type == 'R':
            alphabet = Alphabet(cfg.ADDRESS.ALPHABET)
        else:
            alphabet = Alphabet()

        test_data_dir = cfg.ADDRESS.TEST_DATA_DIR
        test_anno_dir = cfg.ADDRESS.TEST_GT_DIR
        test_set = get_dataset(cfg, name, test_data_dir, test_anno_dir, split='test', alphabet=alphabet)

        test_dataloader = get_dataloader(cfg, name, dataset=test_set, split='test')
        return test_dataloader
