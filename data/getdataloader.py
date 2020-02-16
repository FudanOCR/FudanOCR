'''
getDataLoader函数通过读取配置文件，返回训练和测试的数据集加载器
'''

from alphabet.alphabet import Alphabet
from data import dataset
import torch

def getDataLoader(opt):
    train_root = opt.ADDRESS.RECOGNITION.TRAIN_DATA_DIR
    val_root = opt.ADDRESS.RECOGNITION.TEST_DATA_DIR

    alphabet = Alphabet(opt.ADDRESS.RECOGNITION.ALPHABET)


    train_dataset = dataset.lmdbDataset(root=train_root,
                                        transform=dataset.resizeNormalize((opt.IMAGE.IMG_W, opt.IMAGE.IMG_H)),
                                        reverse=opt.BidirDecoder, alphabet=alphabet.str)
    assert train_dataset

    test_dataset = dataset.lmdbDataset(root=val_root,
                                       transform=dataset.resizeNormalize((opt.IMAGE.IMG_W, opt.IMAGE.IMG_H)),
                                       reverse=opt.BidirDecoder, alphabet=alphabet.str)
    assert test_dataset


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.MODEL.BATCH_SIZE,
        shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.MODEL.BATCH_SIZE),
        num_workers=int(opt.BASE.WORKERS))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.MODEL.BATCH_SIZE,
        shuffle=False,
        num_workers=int(opt.BASE.WORKERS))

    return train_loader, test_loader