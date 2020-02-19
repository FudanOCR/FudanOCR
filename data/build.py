import os
import bisect
import copy
import logging

import torch.utils.data
from . import catalog
from alphabet.alphabet import Alphabet
from data import Imdb_dataset
from data import icdar_seriers
from data import CTW1500
import torch


def get_dataset(cfg, name, data_dir, anno_dir, alphabet):
    if 'Imdb' in name:
        dataset = Imdb.lmdbDataset(root=data_dir,
                                   transform=Imdb_dataset.resizeNormalize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)),
                                   reverse=cfg.BidirDecoder, alphabet=alphabet.str)
        assert dataset
        return dataset

    elif 'ICDAR2013Dataset' in name:
        dataset = icdar_seriers.ICDAR2013Dataset(data_dir, anno_dir)
        assert dataset
        return dataset

    elif 'ICDAR2015TRAIN' in name:
        dataset = icdar_seriers.ICDAR2015TRAIN(data_dir, anno_dir)
        assert dataset
        return dataset

    elif 'CTW1500' in name:
        dataset = CTW1500.CTW1500Loader(data_dir, anno_dir)
    raise RuntimeError("Dataset not available: {}".format(name))


def get_dataloader(cfg, name, dataset):
    if 'Imdb' in name:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.MODEL.BATCH_SIZE,
            shuffle=False, sampler=Imdb_dataset.randomSequentialSampler(train_dataset, cfg.MODEL.BATCH_SIZE),
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
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.MODEL.BATCH_SIZE,
            shuffle=True,
            num_workers=3,
            drop_last=True,
            pin_memory=True)

    raise RuntimeError("Dataset not available: {}".format(name))


def build_dataloader(cfg, is_train=True):
    dataset_name = cfg.DATASETS.DATASET
    # dataset_type = cfg.DATASETS.TYPE
    '''考虑不需区分dataset是否是检测或识别
    if dataset_type == 'DETECTION':
        data_dir = cfg.ADDRESS.DETECTION.TRAIN_DATA_DIR
        anno_dir = cfg.ADDRESS.DETECTION.TRAIN_GT_DIR
    elif dataset_type == 'RECOGNITION':
    '''
    if is_train
        data_dir = cfg.ADDRESS.RECOGNITION.TRAIN_DATA_DIR
        anno_dir = cfg.ADDRESS.RECOGNITION.TRAIN_GT_DIR

        images_per_batch = cfg.MODEL.BATCH_SIZE
        num_gpus = int(cfg.BASE.NUM_GPUS)
        assert (
                images_per_batch % num_gpus == 0
        ), "IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
    else
        data_dir = cfg.ADDRESS.RECOGNITION.TEST_DATA_DIR
        anno_dir = cfg.ADDRESS.RECOGNITION.TEST_GT_DIR

    if images_per_gpu > 5:
        logger = logging.getLogger(__name__)
        logger.warning(
            "每GPU图片数量过高时可能遇到内存溢出，"
            "若发生该情况请调整BATCH_SIZE,并调整学习率等其他可能影响效果的因素"
        )

    alphabet = Alphabet(opt.ADDRESS.RECOGNITION.ALPHABET)
    dataset = get_dataset(cfg, dataset_name, data_dir, anno_dir, alphabet)

    dataloader = get_dataloader(cfg=cfg, name=dataset_name, dataset=dataset)
    return dataloader
