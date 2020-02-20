'''
getDataLoader函数通过读取配置文件，返回训练和测试的数据集加载器
'''

from torch.utils.data import DataLoader
import torch

from model.detection_model.AdvancedEAST.utils.data_utils import custom_dset, collate_fn


def getDataLoader(opt):
    batch_size = opt.MODEL.BATCH_SIZE
    trainset = custom_dset(split='train')
    valset = custom_dset(split='val')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=opt.BASE.WORKERS, drop_last=False)
    val_loader = DataLoader(valset, batch_size=1, collate_fn=collate_fn, num_workers=opt.BASE.WORKERS)

    return train_loader, val_loader
