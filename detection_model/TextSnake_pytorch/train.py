import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.total_text import TotalText
from network.loss import TextLoss
from network.textnet import TextNet
from util.augmentation import EvalTransform, NewAugmentation
from util.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.option import BaseOptions
from util.visualize import visualize_network_output

total_iter = 0


def adjust_learning_rate(optimizer, i):
    if 0 <= i*cfg.batch_size < 100000:
        lr = cfg.lr
    elif 100000 <= i*cfg.batch_size < 400000:
        lr = cfg.lr * 0.1
    else:
        lr = cfg.lr * 0.1 * (0.94 ** ((i*cfg.batch_size-300000) // 100000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_requires_grad(model, i):
    if 0 <= i < 4000:
        for name, param in model.named_parameters():
            if name == 'conv1.0.weight' or name == 'conv1.0.bias' or \
               name == 'conv1.1.weight' or name == 'conv1.1.bias':
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if name == 'conv1.0.weight' or name == 'conv1.0.bias' or \
               name == 'conv1.1.weight' or name == 'conv1.1.bias':
                param.requires_grad = True


def save_model(model, optimizer, scheduler, epoch):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textsnake_{}_{}.pth'.format(model.backbone_name, epoch))
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
        # 'scheduler': scheduler.state_dict()
    }
    torch.save(state_dict, save_path)


def load_model(save_path):
    print('Loading from {}.'.format(save_path))
    checkpoint = torch.load(save_path)
    return checkpoint


def train(model, train_loader, criterion, scheduler, optimizer, epoch, summary_writer):

    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    global total_iter

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        output = model(img)
        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask, total_iter)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss

        # backward
        # scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.viz and i < cfg.vis_num:
            visualize_network_output(output, tr_mask, tcl_mask, prefix='train_{}'.format(i))

        if i % cfg.display_freq == 0:
            print('Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f} - {:.2f}s/step'.format(
                epoch, i, len(train_loader), loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(), cos_loss.item(), radii_loss.item(), batch_time.avg)
                )

        # write summary
        if total_iter % cfg.summary_freq == 0:
            print('Summary in {}'.format(os.path.join(cfg.summary_dir, cfg.exp_name)))
            tr_pred = output[:, 0:2].softmax(dim=1)[:, 1:2]
            tcl_pred = output[:, 2:4].softmax(dim=1)[:, 1:2]
            summary_writer.add_image('input_image', vutils.make_grid(img, normalize=True), total_iter)
            summary_writer.add_image('tr/tr_pred', vutils.make_grid(tr_pred * 255, normalize=True), total_iter)
            summary_writer.add_image('tr/tr_mask', vutils.make_grid(torch.unsqueeze(tr_mask * train_mask, 1) * 255), total_iter)
            summary_writer.add_image('tcl/tcl_pred', vutils.make_grid(tcl_pred * 255, normalize=True), total_iter)
            summary_writer.add_image('tcl/tcl_mask', vutils.make_grid(torch.unsqueeze(tcl_mask * train_mask, 1) * 255), total_iter)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], total_iter)
            summary_writer.add_scalar('model/tr_loss', tr_loss.item(), total_iter)
            summary_writer.add_scalar('model/tcl_loss', tcl_loss.item(), total_iter)
            summary_writer.add_scalar('model/sin_loss', sin_loss.item(), total_iter)
            summary_writer.add_scalar('model/cos_loss', cos_loss.item(), total_iter)
            summary_writer.add_scalar('model/radii_loss', radii_loss.item(), total_iter)
            summary_writer.add_scalar('model/loss', loss.item(), total_iter)

        total_iter += 1

    print('Speed: {}s /step, {}s /epoch'.format(batch_time.avg, time.time() - start))

    if epoch % cfg.save_freq == 0:
        save_model(model, optimizer, scheduler, epoch)

    print('Training Loss: {}'.format(losses.avg))


def validation(model, valid_loader, criterion):

    model.eval()
    losses = AverageMeter()

    for i, (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta) in enumerate(valid_loader):
        print(meta['image_id'])
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map = to_device(
            img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map)

        output = model(img)

        tr_loss, tcl_loss, sin_loss, cos_loss, radii_loss = \
            criterion(output, tr_mask, tcl_mask, sin_map, cos_map, radius_map, train_mask)
        loss = tr_loss + tcl_loss + sin_loss + cos_loss + radii_loss
        losses.update(loss.item())

        if cfg.viz and i < cfg.vis_num:
            visualize_network_output(output, tr_mask, tcl_mask, prefix='val_{}'.format(i))

        if i % cfg.display_freq == 0:
            print(
                'Validation: - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - sin_loss: {:.4f} - cos_loss: {:.4f} - radii_loss: {:.4f}'.format(
                    loss.item(), tr_loss.item(), tcl_loss.item(), sin_loss.item(),
                    cos_loss.item(), radii_loss.item())
            )

    print('Validation Loss: {}'.format(losses.avg))


def main():
    global total_iter
    data_root = os.path.join('/home/shf/fudan_ocr_system/datasets/', cfg.dataset)

    trainset = TotalText(
        data_root=data_root,
        ignore_list=os.path.join(data_root, 'ignore_list.txt'),
        is_training=True,
        transform=NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)
    )

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Model
    model = TextNet(backbone=cfg.backbone, output_channel=7)
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)
    # if cfg.dataset == 'ArT_train':
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 50000], gamma=0.1)
    # elif cfg.dataset == 'LSVT_full_train':
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 50000], gamma=0.1)

    # load model if resume
    if cfg.resume is not None:
        global total_iter
        checkpoint = load_model(cfg.resume)
        cfg.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        total_iter = checkpoint['epoch'] * len(train_loader)

    if not os.path.exists(os.path.join(cfg.summary_dir, cfg.exp_name)):
        os.mkdir(os.path.join(cfg.summary_dir, cfg.exp_name))
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.summary_dir, cfg.exp_name))

    print('Start training TextSnake.')


    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        adjust_learning_rate(optimizer, total_iter)
        train(model, train_loader, criterion, None, optimizer, epoch, summary_writer)

    print('End.')


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # main
    main()