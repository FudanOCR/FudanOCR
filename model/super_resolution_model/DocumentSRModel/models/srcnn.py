import numpy as np
from scipy.misc import imsave
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms

from networks.baseblocks import ConvBlock
from dataloader import TrainDataset, DevDataset, TestDataset
from utils.visualizer import Visualizer
from utils.loss import BCE2d
from utils.normalize import norm, denorm, weights_init_normal
from utils.target import PSNR, SSIM, batch_compare_filter, batch_SSIM


USE_GPU = torch.cuda.is_available()


def save_img(img, save_fn=''):
    if not os.path.exists(os.path.split(save_fn)[0]):
        os.makedirs(os.path.split(save_fn)[0])
    if list(img.shape)[0] == 3:
        save_image = img * 255.0
        save_image = save_image.clamp(
            0, 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    else:
        save_image = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    imsave(save_fn, save_image)


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = ConvBlock(3, 64, 9, 1, 4, norm=None, activation='relu')
        self.conv2 = ConvBlock(64, 32, 1, 1, 0, norm=None, activation='relu')
        self.conv3 = ConvBlock(32, 3, 5, 1, 2, norm=None, activation=None)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return F.sigmoid(out)


class Model(object):
    def __init__(self, cfg):
        # parameter init
        self.env = cfg.env
        self.train_dataset = cfg.train_dataset
        self.valid_dataset = cfg.valid_dataset
        self.test_dataset = cfg.test_dataset
        self.data_dir = cfg.data_dir
        self.save_dir = cfg.save_dir

        self.num_threads = int(cfg.num_threads)
        self.num_epochs = int(cfg.num_epochs)
        self.save_epochs = int(cfg.save_epochs)
        self.pretrain_epochs = int(cfg.pretrain_epochs)
        self.batch_size = int(cfg.batch_size)
        self.valid_batch_size = int(cfg.valid_batch_size)
        self.test_batch_size = int(cfg.test_batch_size)
        self.plot_iter = int(cfg.plot_iter)
        self.crop_size = int(cfg.crop_size)
        self.scale_factor = int(cfg.scale_factor)
        self.lr = float(cfg.lr)

    def load_dataset(self, mode='train', random_scale=True, rotate=True, fliplr=True, fliptb=True):
        if mode == 'train':
            train_set = TrainDataset(os.path.join(self.data_dir, self.train_dataset),
                                     crop_size=self.crop_size, scale_factor=self.scale_factor,
                                     random_scale=random_scale, rotate=rotate, fliplr=fliplr, fliptb=fliptb)
            return DataLoader(dataset=train_set, num_workers=self.num_threads,
                              batch_size=self.batch_size, shuffle=True)
        elif mode == 'valid':
            valid_set = DevDataset(os.path.join(
                self.data_dir, self.valid_dataset))
            return DataLoader(dataset=valid_set, num_workers=self.num_threads,
                              batch_size=self.valid_batch_size, shuffle=False)
        elif mode == 'test':
            test_set = TestDataset(os.path.join(
                self.data_dir, self.test_dataset))
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size, shuffle=False)

    def train(self, srcnn_path=None, random_scale=True, rotate=True, fliplr=True, fliptb=True):
        vis = Visualizer(self.env)

        print('================ Loading datasets =================')
        # load training dataset
        print('## Current Mode: Train')
        # train_data_loader = self.load_dataset(mode='valid')
        train_data_loader = self.load_dataset(
            mode='train', random_scale=random_scale, rotate=rotate, fliplr=fliplr, fliptb=fliptb)

        ##########################################################
        ##################### build network ######################
        ##########################################################
        print('Building Networks and initialize parameters\' weights....')
        # init srnet
        srcnn = SRCNN()
        srcnn.apply(weights_init_normal)

        # load pretrained srresnet or just initialize
        if srcnn_path is None or not os.path.exists(srcnn_path):
            print('===> initialize the srcnn')
            print('======> No pretrained model')
        else:
            print('======> loading the weight from pretrained model')
            pretrained_dict = torch.load(srcnn_path)
            model_dict = srcnn.state_dict()

            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            srcnn.load_state_dict(model_dict)

        # optimizer init
        # different learning rate
        lr = self.lr

        srcnn_optimizer = optim.Adam(srcnn.parameters(), lr=lr, betas=(0.9, 0.999))

        # loss function init
        MSE_loss = nn.MSELoss()
        BCE_loss = nn.BCELoss()

        # cuda accelerate
        if USE_GPU:
            srcnn.cuda()
            MSE_loss.cuda()
            BCE_loss.cuda()
            print('\tCUDA acceleration is available.')

        ##########################################################
        ##################### train network ######################
        ##########################################################
        import torchnet as tnt
        from tqdm import tqdm
        from PIL import Image

        total_avg_loss = tnt.meter.AverageValueMeter()
        psnr_2x_avg = tnt.meter.AverageValueMeter()
        ssim_2x_avg = tnt.meter.AverageValueMeter()
        psnr_4x_avg = tnt.meter.AverageValueMeter()
        ssim_4x_avg = tnt.meter.AverageValueMeter()

        srcnn.train()
        itcnt = 0
        for epoch in range(self.num_epochs):
            psnr_2x_avg.reset()
            ssim_2x_avg.reset()
            psnr_4x_avg.reset()
            ssim_4x_avg.reset()

            # learning rate is decayed by a factor every 20 epoch
            if (epoch + 1 % 20) == 0:
                for param_group in srcnn_optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay for srcnn: lr={}".format(
                    srcnn_optimizer.param_groups[0]["lr"]))

            itbar = tqdm(enumerate(train_data_loader))
            for ii, (hr, lr2x, lr4x, bc2x, bc4x) in itbar:

                mini_batch = hr.size()[0]

                hr_ = Variable(hr)
                lr2x_ = Variable(lr2x)
                lr4x_ = Variable(lr4x)
                bc2x_ = Variable(bc2x)
                bc4x_ = Variable(bc4x)

                # cuda mode setting
                if USE_GPU:
                    hr_ = hr_.cuda()
                    lr2x_ = lr2x_.cuda()
                    lr4x_ = lr4x_.cuda()
                    bc2x_ = bc2x_.cuda()
                    bc4x_ = bc4x_.cuda()

                # =============================================================== #
                # ======================= srcnn training ======================== #
                # =============================================================== #
                sr4x_ = srcnn(bc4x_)

                #============ calculate 4x loss ==============#
                srcnn_optimizer.zero_grad()

                #### Content Loss ####
                content_loss_4x = MSE_loss(sr4x_, hr_)

                #============ calculate scores ==============#
                psnr_4x_score_process = batch_compare_filter(
                    sr4x_.cpu().data, hr, PSNR)
                psnr_4x_avg.add(psnr_4x_score_process)

                ssim_4x_score_process = batch_compare_filter(
                    sr4x_.cpu().data, hr, SSIM)
                ssim_4x_avg.add(ssim_4x_score_process)

                #============== loss backward ===============#
                total_loss_4x = content_loss_4x

                total_loss_4x.backward()
                srcnn_optimizer.step()

                total_avg_loss.add(total_loss_4x.data.item())

                if (ii+1) % self.plot_iter == self.plot_iter-1:
                    res = {'generate loss': total_avg_loss.value()[0]}
                    vis.plot_many(res, 'SRCNN Loss')

                    psnr_4x_score_origin = batch_compare_filter(bc4x, hr, PSNR)
                    res_psnr = {'4x_origin_psnr': psnr_4x_score_origin,
                                '4x_sr_psnr': psnr_4x_score_process}
                    vis.plot_many(res_psnr, 'PSNR Score')

                    ssim_4x_score_origin = batch_compare_filter(bc4x, hr, SSIM)
                    res_ssim = {'4x_origin_ssim': ssim_4x_score_origin,
                                '4x_sr_ssim': ssim_4x_score_process}
                    vis.plot_many(res_ssim, 'SSIM Score')

                #======================= Output result of total training processing =======================#
                itcnt += 1
                itbar.set_description("Epoch: [%2d] [%d/%d] PSNR_2x_Avg: %.6f, SSIM_2x_Avg: %.6f, PSNR_4x_Avg: %.6f, SSIM_4x_Avg: %.6f"
                                      % ((epoch + 1), (ii + 1), len(train_data_loader),
                                         psnr_2x_avg.value()[0], ssim_2x_avg.value()[
                                          0],
                                         psnr_4x_avg.value()[0], ssim_4x_avg.value()[0]))

                if (ii+1) % self.plot_iter == self.plot_iter-1:

                    vis.images(lr4x, win='LR4X image',
                               opts=dict(title='LR4X image'))
                    vis.images(bc4x, win='BC4X image',
                               opts=dict(title='BC4X image'))
                    vis.images(sr4x_.cpu().data, win='SR4X image',
                               opts=dict(title='SR4X image'))

                    vis.images(hr, win='HR image',
                               opts=dict(title='HR image'))

            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(srcnn, os.path.join(self.save_dir, 'checkpoints', 'srcnn'), 'srcnn_param_batch{}_lr{}_epoch{}'.
                                format(self.batch_size, self.lr, epoch+1))

        # Save final trained model and results
        vis.save([self.env])
        self.save_model(srcnn, os.path.join(self.save_dir, 'checkpoints', 'srcnn'), 'srcnn_param_batch{}_lr{}_epoch{}'.
                        format(self.batch_size, self.lr, self.num_epochs))

    def test(self, srcnn_path=None):
        test_data_dir = os.path.join(self.data_dir, self.test_dataset)
        result_data_dir = os.path.join(self.save_dir, "test_results", "SRCNN_"+self.test_dataset)
        if not os.path.exists(result_data_dir):
            os.makedirs(result_data_dir)

        # judge whether model exists
        if not os.path.exists(srcnn_path):
            raise Exception('srcnn model not exists')

        # load network params
        srcnn = SRCNN()
        srcnn.load_state_dict(torch.load(srcnn_path))

        if USE_GPU:
            srcnn.cuda()

        import torchnet as tnt
        from tqdm import tqdm
        from PIL import Image

        psnr_4x_avg = tnt.meter.AverageValueMeter()
        ssim_4x_avg = tnt.meter.AverageValueMeter()

        psnr_4x_bc_avg = tnt.meter.AverageValueMeter()
        ssim_4x_bc_avg = tnt.meter.AverageValueMeter()

        srcnn.eval()

        # processing test data
        iterbar = tqdm(os.listdir(test_data_dir))
        for img_name in iterbar:
            img = Image.open(os.path.join(test_data_dir, img_name)).convert("RGB")
            # transform = Transforms.RandomCrop(self.crop_size)
            # img = transform(img)

            w, h = img.size[0], img.size[1]
            w_lr4x, h_lr4x = int(
                w // self.scale_factor), int(h // self.scale_factor)
            w_lr2x, h_lr2x = w_lr4x * 2, h_lr4x * 2
            w_hr, h_hr = w_lr4x * self.scale_factor, h_lr4x * self.scale_factor

            # transform tensor
            hr = img.resize((w_hr, h_hr), Image.ANTIALIAS)
            lr2x = img.resize((w_lr2x, h_lr2x), Image.ANTIALIAS)
            bc2x = lr2x.resize((w_hr, h_hr), Image.BICUBIC)
            lr4x = img.resize((w_lr4x, h_lr4x), Image.ANTIALIAS)
            bc4x = lr4x.resize((w_hr, h_hr), Image.BICUBIC)

            hr_ = Transforms.ToTensor()(hr).unsqueeze(0)
            lr2x_ = Transforms.ToTensor()(lr2x).unsqueeze(0)
            bc2x_ = Transforms.ToTensor()(bc2x).unsqueeze(0)
            lr4x_ = Transforms.ToTensor()(lr4x).unsqueeze(0)
            bc4x_ = Transforms.ToTensor()(bc4x).unsqueeze(0)

            if USE_GPU:
                hr_ = hr_.cuda()
                lr2x_ = lr2x_.cuda()
                bc2x_ = bc2x_.cuda()
                lr4x_ = lr4x_.cuda()
                bc4x_ = bc4x_.cuda()

            sr4x_ = srcnn(bc2x_)

            # calculate PSNR & SSIM
            psnr_4x_score = batch_compare_filter(
                sr4x_.cpu().data, hr_, PSNR)
            ssim_4x_score = batch_compare_filter(
                sr4x_.cpu().data, hr_, SSIM)
            psnr_4x_avg.add(psnr_4x_score)
            ssim_4x_avg.add(ssim_4x_score)

            psnr_4x_score = batch_compare_filter(
                bc2x_, hr_, PSNR)
            ssim_4x_score = batch_compare_filter(
                bc2x_, hr_, SSIM)
            psnr_4x_bc_avg.add(psnr_4x_score)
            ssim_4x_bc_avg.add(ssim_4x_score)

            # save image
            save_img(sr4x_.cpu().data, os.path.join(result_data_dir, img_name))

        print("final PSNR score: {}".format(psnr_4x_bc_avg.value()[0]))
        print("final SSIM score: {}".format(ssim_4x_bc_avg.value()[0]))
        print("final PSNR score: {}".format(psnr_4x_avg.value()[0]))
        print("final SSIM score: {}".format(ssim_4x_avg.value()[0]))

    def save_model(self, model, save_dir, model_name, mtype='pkl'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if mtype == 'pkl':
            save_path = os.path.join(save_dir, model_name+'.pkl')
            torch.save(model.state_dict(), save_path)
        elif mtype == 'pth':
            save_path = os.path.join(save_dir, model_name+'.pth')
            torch.save(model.state_dict(), save_path)
