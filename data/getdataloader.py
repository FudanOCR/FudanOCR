'''
getDataLoader函数通过读取配置文件，返回训练和测试的数据集加载器
'''

from alphabet.alphabet import Alphabet
# from data import dataset
from data.total_text import TotalText
from model.detection_model.TextSnake_pytorch.util.augmentation import EvalTransform, NewAugmentation
import torch
import os

def getDataLoader(opt):
    train_root = opt.ADDRESS.TRAIN_DATA_DIR
    val_root = opt.ADDRESS.VAL_DATA_DIR


    train_dataset = TotalText(
            data_root=train_root,
            ignore_list=os.path.join(train_root, 'ignore_list.txt'),
            is_training=True,
            transform=NewAugmentation(size=opt.TEXTSNAKE.input_size, mean=opt.TEXTSNAKE.means, std=opt.TEXTSNAKE.stds, maxlen=1280, minlen=512)
    )
    assert train_dataset

    test_dataset = TotalText(
            data_root=val_root,
            ignore_list=os.path.join(train_root, 'ignore_list.txt'),
            is_training=False,
            transform=EvalTransform(size=1280, mean=opt.TEXTSNAKE.means, std=opt.TEXTSNAKE.stds),
            #NewAugmentation(size=opt.TEXTSNAKE.input_size, mean=opt.TEXTSNAKE.means, std=opt.TEXTSNAKE.stds, maxlen=1280, minlen=512)
    )
    assert test_dataset

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.MODEL.BATCH_SIZE, shuffle=True, num_workers=opt.BASE.WORKERS)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.MODEL.BATCH_SIZE,
        shuffle=False,
        num_workers=int(opt.BASE.WORKERS))

    return train_loader, test_loader