# -*- coding: utf-8 -*-
def train_AEAST(config_file):

    import sys
    sys.path.append('./detection_model/AdvancedEAST')

    import os
    import argparse
    import time
    import numpy as numpy
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    from tqdm import tqdm

    import config as cfg
    from utils.data_utils import custom_dset, collate_fn
    from network.AEast import East
    from network.loss import LossFunc
    from utils.utils import AverageMeter, save_log
    from utils.earlystop import EarlyStopping

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)  # 获取了yaml文件
    '''
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--valpre', action='store_true', help='Validate pretrained model before training')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--pretrain', '-p', action='store_true', help='Enable pretrain.')
        group.add_argument('--resume', '-r', action='store_true', help='Resume training.')
        return parser.parse_args()
    '''
    class Wrapped:
        def __init__(self, train_loader, val_loader, model, criterion, optimizer, scheduler, start_epoch, val_loss_min):
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.start_epoch = start_epoch
            self.tick = time.strftime("%Y%m%d-%H-%M-%S", time.localtime(time.time()))
            self.earlystopping = EarlyStopping(opt.patience, val_loss_min)
            # self.valpre = valpre

        def __call__(self):
            for epoch in tqdm(range(self.start_epoch + 1, opt.max_epoch + 1), desc='Epoch'):
                # if epoch == 1 and self.valpre:
                if epoch == 1:
                    tqdm.write("Validating pretrained model.")
                    self.validate(0)
                if epoch > 1 and epoch % opt.decay_step == 0:
                    tqdm.write(f"Learning rate - Epoch: [{epoch - 1}]: {self.optimizer.param_groups[0]['lr']:.6f}")
                self.train(epoch)
                if self.validate(epoch):  # if earlystop
                    print('Earlystopping activates. Training stopped.')
                    break

        def validate(self, epoch):
            losses = AverageMeter()

            self.model.eval()
            for i, (img, gt) in tqdm(enumerate(self.val_loader), desc='Val', total=len(self.val_loader)):
                img = img.cuda()
                gt = gt.cuda()
                east_detect = self.model(img)
                loss = self.criterion(gt, east_detect)
                losses.update(loss.item(), img.size(0))
            tqdm.write(f'Validate Loss - Epoch: [{epoch}]  Avg Loss {losses.avg:.4f}')
            save_log(losses, epoch, i + 1, len(self.val_loader), self.tick, split='Validation')

            earlystop, save = self.earlystopping(losses.avg)
            if not earlystop and save:
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'val_loss_min': losses.avg
                }
                self.earlystopping.save_checkpoint(state, losses.avg)
            return earlystop

        def train(self, epoch):
            losses = AverageMeter()

            self.model.train()
            for i, (img, gt) in tqdm(enumerate(self.train_loader), desc='Train', total=len(self.train_loader)):
                img = img.cuda()
                gt = gt.cuda()
                east_detect = self.model(img)
                loss = self.criterion(gt, east_detect)
                losses.update(loss.item(), img.size(0))

                # backward propagation
                self.scheduler.step()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % opt.print_step == 0:
                    tqdm.write(
                        'Training loss - Epoch: [{0}][{1}/{2}] Loss {loss.val:.4f} Avg Loss {loss.avg:.4f}'.format(
                            epoch, i + 1, len(self.train_loader), loss=losses))
                save_log(losses, epoch, i + 1, len(self.train_loader), self.tick, split='Training')

    class LRPolicy:
        def __init__(self, rate, step):
            self.rate = rate
            self.step = step

        def __call__(self, it):
            return self.rate ** (it // self.step)

    # args = parse_args()
    print('=== AdvancedEAST ===')
    print(f'Task id: {opt.task_id}')
    print('=== Initialzing DataLoader ===')
    print(f'Multi-processing on {opt.num_process} cores')
    # batch_size = 1
    # batch_size = opt.batch_size_per_gpu * len(opt.gpu_ids)
    batch_size = opt.batch_size_per_gpu

    # trainset = custom_dset(config_file, split='train')
    # valset = custom_dset(config_file, split='val')
    trainset = custom_dset(split='train')
    valset = custom_dset(split='val')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=opt.num_workers, drop_last=False)
    val_loader = DataLoader(valset, batch_size=1, collate_fn=collate_fn, num_workers=opt.num_workers)

    print('=== Building Network ===')
    model = East()
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[1,2])
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    # model = nn.DataParallel(model, device_ids=[0,1,2])
    # model = nn.DataParallel(model)
    model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {params}')
    '''
    if args.pretrain:
        print('=== Loading Pretrained Model ===')
        assert os.path.isfile(opt.pre_file), 'Pretrained model does not exist.'
        print(f'Loading {opt.pre_file}')
        checkpoint = torch.load(opt.pre_file)
        model.module.load_state_dict(checkpoint['state_dict'])
    '''
    cudnn.benchmark = True
    criterion = LossFunc()
    optimizer = Adam(model.parameters(), lr=opt.lr_rate)

    # decay every opt.decay_step epoch / every decay_step iter
    decay_step = len(train_loader) * opt.decay_step
    scheduler = LambdaLR(optimizer, lr_lambda=LRPolicy(rate=opt.decay_rate, step=decay_step))
    print(f'Batch size: {batch_size}')
    print('Initial learning rate: {0}\nDecay step: {1}\nDecay rate: {2}\nPatience: {3}'.format(
        opt.lr_rate, opt.decay_step, opt.decay_rate, opt.patience))
    '''
    if args.resume:  # if resume
        print('=== Loading Checkpoint ===')
        cp_file = opt.task_id + '_best.pth.tar'
        cp_path = os.path.join(opt.result_dir, cp_file)
        assert os.path.isfile(cp_path), 'Checkpoint file does not exist.'
        print(f'Loading {cp_path}')

        checkpoint = torch.load(cp_path)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        val_loss_min = checkpoint['val_loss_min']
        print(f"Start epoch set to {start_epoch + 1}")
        print(f"Learning rate set to {optimizer.param_groups[0]['lr']:.6f}")
    else:
        start_epoch = 0
        val_loss_min = None
    '''
    start_epoch = 0
    val_loss_min = None

    print('=== Training ===')
    wrap = Wrapped(train_loader, val_loader, model, criterion, optimizer, scheduler, start_epoch, val_loss_min)
    wrap()

