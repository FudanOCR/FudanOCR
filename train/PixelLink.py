def train_PixelLink(config_file):

    import sys
    sys.path.append('./detection_model/PixelLink')

    import net
    import numpy as np
    import torch
    import torch.nn as nn
    # import read_data
    import datasets
    from torch import optim
    from criterion import PixelLinkLoss
    import loss
    import config
    import postprocess
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
    import os
    import cv2
    import time
    import argparse
    import ImgLib.ImgShow as ImgShow
    import ImgLib.ImgTransform as ImgTransform
    # import moduletest.test_postprocess as test_postprocess
    from test_model import test_on_train_dataset

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)  # 获取了yaml文件
    

    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


    def train(epoch, iteration, dataloader, my_net, optimizer, optimizer2, device):
        for i in range(epoch):
            for i_batch, sample in enumerate(dataloader):
                start = time.time()
                images = sample['image'].to(device)
                # print(images.shape, end=" ")
                pixel_masks = sample['pixel_mask'].to(device)
                neg_pixel_masks = sample['neg_pixel_mask'].to(device)
                link_masks = sample['link_mask'].to(device)
                pixel_pos_weights = sample['pixel_pos_weight'].to(device)

                out_1, out_2 = my_net.forward(images)
                loss_instance = PixelLinkLoss()
                # print(out_2)

                pixel_loss_pos, pixel_loss_neg = loss_instance.pixel_loss(out_1, pixel_masks, neg_pixel_masks,
                                                                          pixel_pos_weights)
                pixel_loss = pixel_loss_pos + pixel_loss_neg
                link_loss_pos, link_loss_neg = loss_instance.link_loss(out_2, link_masks)
                link_loss = link_loss_pos + link_loss_neg
                losses = opt.pixel_weight * pixel_loss + opt.link_weight * link_loss
                print("iteration %d" % iteration, end=": ")
                print("pixel_loss: " + str(pixel_loss.tolist()), end=", ")
                # print("pixel_loss_pos: " + str(pixel_loss_pos.tolist()), end=", ")
                # print("pixel_loss_neg: " + str(pixel_loss_neg.tolist()), end=", ")
                print("link_loss: " + str(link_loss.tolist()), end=", ")
                # print("link_loss_pos: " + str(link_loss_pos.tolist()), end=", ")
                # print("link_loss_neg: " + str(link_loss_neg.tolist()), end=", ")
                print("total loss: " + str(losses.tolist()), end=", ")
                if iteration < 100:
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                else:
                    optimizer2.zero_grad()
                    losses.backward()
                    optimizer2.step()
                end = time.time()
                print("time: " + str(end - start))
                if (iteration + 1) % 200 == 0:
                    # if args.change:
                    #     saving_model_dir = opt.saving_model_dir3
                    # else:
                    saving_model_dir = opt.saving_model_dir
                    torch.save(my_net.state_dict(), saving_model_dir + str(iteration + 1) + ".mdl")
                iteration += 1

    def main():
        dataset = datasets.PixelLinkIC15Dataset(opt.train_images_dir, opt.train_labels_dir)
        sampler = WeightedRandomSampler([1 / len(dataset)] * len(dataset), opt.batch_size, replacement=True)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler)
        # dataloader = DataLoader(dataset, opt.batch_size, shuffle=True)
        my_net = net.Net()

        if opt.gpu:
            device = torch.device("cuda:0")
            my_net = my_net.cuda()
            if opt.multi_gpu:
                my_net = nn.DataParallel(my_net)
        else:
            device = torch.device("cpu")

        # nn.init.xavier_uniform_(list(my_net.parameters()))
        my_net.apply(weight_init)
        optimizer = optim.SGD(my_net.parameters(), lr=opt.learning_rate, momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        # if args.change:
        optimizer2 = optim.SGD(my_net.parameters(), lr=opt.learning_rate2, momentum=opt.momentum,
                               weight_decay=opt.weight_decay)
        # else:
        #     optimizer2 = optim.SGD(my_net.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

        iteration = 0
        train(opt.epoch, iteration, dataloader, my_net, optimizer, optimizer2, device)

    main()

