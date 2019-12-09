# -*- coding:utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

import sys
sys.path.append('/home/cjy/FudanOCR/recognition_model/GRCNN')

import argparse
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import Levenshtein
from torch.autograd import Variable
# from warpctc_pytorch import CTCLoss
from utils.Logger import Logger

import utils.keys as keys
import utils.util as util
import dataset
import models.crann as crann
import yaml
import os
import time

# nohup python3 -u crann_main.py >>lsvt_svhn.out &
# python3 /workspace/mnt/group/ocr/zhangpeiyao/zhang/CRNN/zhangpy/crann_main.py

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--yaml',
                    default='/workspace/mnt/group/ocr/zhangpeiyao/zhang/CRNN/zhangpy/config/icdar/grcnn_art.yml',
                    help='path to config yaml')


def adjust_lr(optimizer, base_lr, epoch, step):
    lr = base_lr * (0.1 ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, val_loader, criterion, optimizer, opt, converter, epoch, logger):
    # Set up training phase.
    interval = int(len(train_loader) / opt['SAVE_FREQ'])
    model.train()

    for i, (cpu_images, cpu_gt) in enumerate(train_loader, 1):
        # print('iter {} ...'.format(i))
        bsz = cpu_images.size(0)
        text, text_len = converter.encode(cpu_gt)
        v_images = Variable(cpu_images.cuda())
        v_gt = Variable(text)
        v_gt_len = Variable(text_len)

        model = model.cuda()
        predict = model(v_images)
        predict_len = Variable(torch.IntTensor([predict.size(0)] * bsz))

        loss = criterion(predict, v_gt, predict_len, v_gt_len)
        logger.scalar_summary('train_loss', loss.data[0], i + epoch * len(train_loader))

        # Compute accuracy
        _, acc = predict.max(2)
        acc = acc.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(acc.data, predict_len.data, raw=False)
        n_correct = 0
        for pred, target in zip(sim_preds, cpu_gt):
            if pred.lower() == target.lower():
                n_correct += 1
        accuracy = n_correct / float(bsz)

        logger.scalar_summary('train_accuray', accuracy, i + epoch * len(train_loader))

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % interval == 0 and i > 0:
            print('Training @ Epoch: [{0}][{1}/{2}]; Train Accuracy:{3}'.format(epoch, i, len(train_loader), accuracy))
            val(model, val_loader, criterion, converter, epoch, i + epoch * len(train_loader), logger, False)
            model.train()
            freq = int(i / interval)
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            '{0}/crann_{1}_{2}.pth'.format(opt['SAVE_PATH'], epoch, freq))


def val(model, ds_loader, criterion, converter, epoch, iteration, logger, valonly):
    print('Start validating on epoch:{0}/iter:{1}...'.format(epoch, iteration))
    model.eval()
    ave_loss = 0.0
    ave_accuracy = 0.0
    err_sim = []
    err_gt = []
    distance = 0
    length = 0
    with torch.no_grad():
        for i, (cpu_images, cpu_gt) in enumerate(ds_loader):
            bsz = cpu_images.size(0)
            text, text_len = converter.encode(cpu_gt)
            v_Images = Variable(cpu_images.cuda())
            v_gt = Variable(text)
            v_gt_len = Variable(text_len)

            predict = model(v_Images)
            predict_len = Variable(torch.IntTensor([predict.size(0)] * bsz))
            loss = criterion(predict, v_gt, predict_len, v_gt_len)
            ave_loss += loss.data[0]

            # Compute accuracy
            _, acc = predict.max(2)
            acc = acc.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(acc.data, predict_len.data, raw=False)
            n_correct = 0
            for pred, target in zip(sim_preds, cpu_gt):
                length += len(target)
                if pred.lower() == target.lower():
                    n_correct += 1.0
                else:
                    err_sim.append(pred)
                    err_gt.append(target)
            ave_accuracy += n_correct / float(bsz)
        for pred, gt in zip(err_sim, err_gt):
            print('pred: %-20s, gt: %-20s' % (pred, gt))
            distance += Levenshtein.distance(pred, gt)
        # print("The Levenshtein distance is:",distance)
        print("The average Levenshtein distance is:", distance / length)
        if not valonly:
            logger.scalar_summary('validation_loss', ave_loss / len(ds_loader), iteration)
            logger.scalar_summary('validation_accuracy', ave_accuracy / len(ds_loader), iteration)
            logger.scalar_summary('Ave_Levenshtein_distance', distance / length, iteration)
        print('Testing Accuracy:{0}, Testing Loss:{1} @ Epoch{2}, Iteration{3}'.format(ave_accuracy / len(ds_loader),
                                                                                       ave_loss / len(ds_loader),
                                                                                       epoch, iteration))


def save_checkpoint(state, file_name):
    # time.sleep(0.01)
    # torch.save(state, file_name)
    try:
        time.sleep(0.01)
        torch.save(state, file_name)
    except RuntimeError:
        print("RuntimeError")
        pass


def train_grcnn(config_yaml):
    '''
    Training/Finetune CNN_RNN_Attention Model.
    '''
    #### Load config settings. ####
    f = open(config_yaml, encoding='utf-8')
    opt = yaml.load(f)
    if os.path.isdir(opt['LOGGER_PATH']) == False:
        os.mkdir(opt['LOGGER_PATH'])
    logger = Logger(opt['LOGGER_PATH'])
    if os.path.isdir(opt['SAVE_PATH']) == False:
        os.system('mkdir -p {0}'.format(opt['SAVE_PATH']))
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True

    #### Set up DataLoader. ####
    train_cfg = opt['TRAIN']
    ds_cfg = train_cfg['DATA_SOURCE']
    print('Building up dataset:{}'.format(ds_cfg['TYPE']))
    if ds_cfg['TYPE'] == 'SYN_DATA':
        text_gen = util.TextGenerator(ds_cfg['GEN_SET'], ds_cfg['GEN_LEN'])
        ds_train = dataset.synthDataset(ds_cfg['FONT_ROOT'], ds_cfg['FONT_SIZE'], text_gen)
    elif ds_cfg['TYPE'] == 'IMG_DATA':
        ds_train = dataset.trainDataset(ds_cfg['IMG_ROOT'], ds_cfg['TRAIN_SET'],
                                        transform=None)  # dataset.graybackNormalize()
    assert ds_train
    train_loader = torch.utils.data.DataLoader(ds_train,
                                               batch_size=train_cfg['BATCH_SIZE'],
                                               shuffle=True,
                                               sampler=None,
                                               num_workers=opt['WORKERS'],
                                               collate_fn=dataset.alignCollate(
                                                   imgH=train_cfg['IMG_H'],
                                                   imgW=train_cfg['MAX_W']))

    val_cfg = opt['VALIDATION']
    ds_val = dataset.testDataset(val_cfg['IMG_ROOT'], val_cfg['VAL_SET'], transform=None)  # dataset.graybackNormalize()
    assert ds_val
    val_loader = torch.utils.data.DataLoader(ds_val,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=opt['WORKERS'],
                                             collate_fn=dataset.alignCollate(
                                                 imgH=train_cfg['IMG_H'],
                                                 imgW=train_cfg['MAX_W']))

    #### Model construction and Initialization. ####
    alphabet = keys.alphabet
    nClass = len(alphabet) + 1

    if opt['N_GPU'] > 1:
        opt['RNN']['multi_gpu'] = True
    else:
        opt['RNN']['multi_gpu'] = False
    model = crann.CRANN(opt, nClass)
    # print(model)

    #### Train/Val the model. ####
    converter = util.strLabelConverter(alphabet)
    from warpctc_pytorch import CTCLoss
    criterion = CTCLoss()
    if opt['CUDA']:
        model.cuda()
        criterion.cuda()

    if opt['OPTIMIZER'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=opt['TRAIN']['LR'])
    elif opt['OPTIMIZER'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt['TRAIN']['LR'],
                               betas=(opt['TRAIN']['BETA1'], 0.999))
    elif opt['OPTIMIZER'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt['TRAIN']['LR'])
    else:
        optimizer = optim.Adadelta(model.parameters(), lr=opt['TRAIN']['LR'])

    start_epoch = 0
    if opt['VAL_ONLY']:
        print('=>loading pretrained model from %s for val only.' % opt['CRANN'])
        checkpoint = torch.load(opt['CRANN'])
        model.load_state_dict(checkpoint['state_dict'])
        val(model, val_loader, criterion, converter, 0, 0, logger, True)
    elif opt['FINETUNE']:
        print('=>loading pretrained model from %s for finetuen.' % opt['CRANN'])
        checkpoint = torch.load(opt['CRANN'])
        # model.load_state_dict(checkpoint['state_dict'])
        model_dict = model.state_dict()
        # print(model_dict.keys())
        cnn_dict = {"cnn." + k: v for k, v in checkpoint.items() if "cnn." + k in model_dict}
        model_dict.update(cnn_dict)
        model.load_state_dict(model_dict)
        for epoch in range(start_epoch, opt['EPOCHS']):
            adjust_lr(optimizer, opt['TRAIN']['LR'], epoch, opt['STEP'])
            train(model, train_loader, val_loader, criterion, optimizer, opt, converter, epoch, logger)
    elif opt['RESUME']:
        print('=>loading checkpoint from %s for resume training.' % opt['CRANN'])
        checkpoint = torch.load(opt['CRANN'])
        start_epoch = checkpoint['epoch'] + 1
        print('resume from epoch:{}'.format(start_epoch))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for epoch in range(start_epoch, opt['EPOCHS']):
            adjust_lr(optimizer, opt['TRAIN']['LR'], epoch, opt['STEP'])
            train(model, train_loader, val_loader, criterion, optimizer, opt, converter, epoch, logger)
    else:
        print('train from scratch.')
        for epoch in range(start_epoch, opt['EPOCHS']):
            adjust_lr(optimizer, opt['TRAIN']['LR'], epoch, opt['STEP'])
            train(model, train_loader, val_loader, criterion, optimizer, opt, converter, epoch, logger)


if __name__ == '__main__':
    opt = parser.parse_args()
    train_grcnn(opt.yaml)

