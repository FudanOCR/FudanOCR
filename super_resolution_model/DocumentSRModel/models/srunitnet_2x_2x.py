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

from dataloader import TrainDataset, DevDataset, TestDataset
from networks.unet import UNet, unet_weight_init
from networks.hed import HED, HED_1L, hed_weight_init
from networks.resnet import ResnetGenerator, Upscale4xResnetGenerator, Upscale2xResnetGenerator
from networks.resnet_wdsr import WDSRResnetGenerator
from networks.discriminators import NLayerDiscriminator
from networks.vggfeature import VGGFeatureMap
from utils.visualizer import Visualizer
from utils.loss import BCE2d
from utils.normalize import norm, denorm, weights_init_normal
from utils.target import PSNR, SSIM, batch_compare_filter, batch_SSIM


USE_GPU = torch.cuda.is_available()
NORM = 'batch'

from scipy.misc import imsave
def save_img(img, save_fn=''):
    if not os.path.exists(os.path.split(save_fn)[0]):
        os.makedirs(os.path.split(save_fn)[0])
    if list(img.shape)[0] == 3:
        # save_image = img * 125.0
        save_image = img
        save_image = save_image.clamp(0, 1).numpy().transpose(1, 2, 0)
    else:
        save_image = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    imsave(save_fn, save_image)


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
                              batch_size=self.valid_batch_size, shuffle=True)
        elif mode == 'test':
            test_set = TestDataset(os.path.join(
                self.data_dir, self.test_dataset))
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size, shuffle=False)

    def train(self, edgenetpath=None, sr2x1_path=None, sr2x2_path=None, srcnn_path=None, srresnet_path=None,
            is_fine_tune=False, random_scale=True, rotate=True, fliplr=True, fliptb=True):
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
        # init sr resnet
        # srresnet2x1 = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
        #                                        norm=NORM, activation='prelu', learn_residual=True)
        # srresnet2x2 = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
        #                                        norm=NORM, activation='prelu',learn_residual=True)
        srresnet2x1 = WDSRResnetGenerator(input_nc=3, output_nc=3, n_blocks=5)
        srresnet2x2 = WDSRResnetGenerator(input_nc=3, output_nc=3, n_blocks=5)
        srresnet2x1.apply(weights_init_normal)
        srresnet2x2.apply(weights_init_normal)

        # init discriminator
        discnet = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=5)

        # init edgenet
        edgenet = HED_1L()
        if edgenetpath is None or not os.path.exists(edgenetpath):
            raise Exception('Invalid edgenet model')
        else:
            pretrained_dict = torch.load(edgenetpath)
            model_dict = edgenet.state_dict()
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            edgenet.load_state_dict(model_dict)

        # init vgg feature
        featuremapping = VGGFeatureMap(models.vgg19(pretrained=True))

        # load pretrained srresnet or just initialize
        if sr2x1_path is None or not os.path.exists(sr2x1_path):
            print('===> initialize the srresnet2x1')
            print('======> No pretrained model')
        else:
            print('======> loading the weight from pretrained model')
            pretrained_dict = torch.load(sr2x1_path)
            model_dict = srresnet2x1.state_dict()

            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            srresnet2x1.load_state_dict(model_dict)

        if sr2x2_path is None or not os.path.exists(sr2x2_path):
            print('===> initialize the srresnet2x2')
            print('======> No pretrained model')
        else:
            print('======> loading the weight from pretrained model')
            pretrained_dict = torch.load(sr2x2_path)
            model_dict = srresnet2x2.state_dict()

            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            srresnet2x2.load_state_dict(model_dict)

        # optimizer init
        # different learning rate
        lr = self.lr

        srresnet2x1_optimizer = optim.Adam(
            srresnet2x1.parameters(), lr=lr, betas=(0.9, 0.999))
        srresnet2x2_optimizer = optim.Adam(
            srresnet2x2.parameters(), lr=lr, betas=(0.9, 0.999))
        disc_optimizer = optim.Adam(
            discnet.parameters(), lr=lr/10, betas=(0.9, 0.999))

        # loss function init
        MSE_loss = nn.MSELoss()
        BCE_loss = nn.BCELoss()

        # cuda accelerate
        if USE_GPU:
            edgenet.cuda()
            srresnet2x1.cuda()
            srresnet2x2.cuda()
            discnet.cuda()
            featuremapping.cuda()
            MSE_loss.cuda()
            BCE_loss.cuda()
            print('\tCUDA acceleration is available.')

        ##########################################################
        ##################### train network ######################
        ##########################################################
        import torchnet as tnt
        from tqdm import tqdm
        from PIL import Image

        # batchnorm = nn.BatchNorm2d(1).cuda()

        edge_avg_loss = tnt.meter.AverageValueMeter()
        total_avg_loss = tnt.meter.AverageValueMeter()
        disc_avg_loss = tnt.meter.AverageValueMeter()
        # psnr_2x_avg = tnt.meter.AverageValueMeter()
        # ssim_2x_avg = tnt.meter.AverageValueMeter()
        # psnr_4x_avg = tnt.meter.AverageValueMeter()
        # ssim_4x_avg = tnt.meter.AverageValueMeter()

        srresnet2x1.train()
        srresnet2x2.train()
        discnet.train()

        itcnt = 0
        for epoch in range(self.num_epochs):
            edge_avg_loss.reset()
            total_avg_loss.reset()
            disc_avg_loss.reset()
            # psnr_2x_avg.reset()
            # ssim_2x_avg.reset()
            # psnr_4x_avg.reset()
            # ssim_4x_avg.reset()

            # learning rate is decayed by a factor every 20 epoch
            if (epoch + 1) % 5 == 0:
                for param_group in srresnet2x1_optimizer.param_groups:
                    param_group["lr"] *= 0.5
                print("Learning rate decay for srresnet2x1: lr={}".format(
                    srresnet2x1_optimizer.param_groups[0]["lr"]))
                for param_group in srresnet2x2_optimizer.param_groups:
                    param_group["lr"] *= 0.5
                print("Learning rate decay for srresnet2x2: lr={}".format(
                    srresnet2x2_optimizer.param_groups[0]["lr"]))
                for param_group in disc_optimizer.param_groups:
                    param_group["lr"] *= 0.5
                print("Learning rate decay for discnet: lr={}".format(
                    disc_optimizer.param_groups[0]["lr"]))

            itbar = tqdm(enumerate(train_data_loader))
            for ii, (hr, lr2x, lr4x, bc2x, bc4x) in itbar:

                mini_batch = hr.size()[0]

                hr_ = Variable(hr)
                lr2x_ = Variable(lr2x)
                lr4x_ = Variable(lr4x)
                bc2x_ = Variable(bc2x)
                bc4x_ = Variable(bc4x)
                real_label = Variable(torch.ones(mini_batch))
                fake_label = Variable(torch.zeros(mini_batch))

                # cuda mode setting
                if USE_GPU:
                    hr_ = hr_.cuda()
                    lr2x_ = lr2x_.cuda()
                    lr4x_ = lr4x_.cuda()
                    bc2x_ = bc2x_.cuda()
                    bc4x_ = bc4x_.cuda()
                    real_label = real_label.cuda()
                    fake_label = fake_label.cuda()

                # =============================================================== #
                # ================ Edge-based srresnet training ================= #
                # =============================================================== #
                sr2x_ = srresnet2x1(lr4x_)
                sr4x_ = srresnet2x2(lr2x_)

                '''===================== Train Discriminator ====================='''
                if epoch + 1 > self.pretrain_epochs:
                    disc_optimizer.zero_grad()

                    #===== 2x disc loss =====#
                    real_decision_2x = discnet(lr2x_)
                    real_loss_2x = BCE_loss(
                        real_decision_2x, real_label.detach())

                    fake_decision_2x = discnet(sr2x_.detach())
                    fake_loss_2x = BCE_loss(
                        fake_decision_2x, fake_label.detach())

                    disc_loss_2x = real_loss_2x + fake_loss_2x

                    disc_loss_2x.backward()
                    disc_optimizer.step()

                    #===== 4x disc loss =====#
                    real_decision_4x = discnet(hr_)
                    real_loss_4x = BCE_loss(
                        real_decision_4x, real_label.detach())

                    fake_decision_4x = discnet(sr4x_.detach())
                    fake_loss_4x = BCE_loss(
                        fake_decision_4x, fake_label.detach())

                    disc_loss_4x = real_loss_4x + fake_loss_4x

                    disc_loss_4x.backward()
                    disc_optimizer.step()

                    disc_avg_loss.add(
                        (disc_loss_2x + disc_loss_4x).data.item())

                '''=================== Train srresnet Generator ==================='''
                edge_trade_off = [0.7, 0.2, 0.1, 0.05, 0.01, 0.3]
                if epoch + 1 > self.pretrain_epochs:
                    a1, a2, a3 = 0.75, 0.1, 0.65
                else:
                    a1, a2, a3 = 0.75, 0.0, 0.7

                if not is_fine_tune:
                    #============ calculate 2x loss ==============#
                    srresnet2x1_optimizer.zero_grad()

                    #### Edgenet Loss ####
                    pred = edgenet(sr2x_)
                    real = edgenet(lr2x_)

                    edge_loss_2x = BCE_loss(pred.detach(), real.detach())
                    # for i in range(6):
                    #     edge_loss_2x += edge_trade_off[i] * \
                    #         BCE_loss(pred[i].detach(), real[i].detach())
                    # edge_loss = 0.7 * BCE2d(pred[0], real[i]) + 0.3 * BCE2d(pred[5], real[i])

                    #### Content Loss ####
                    content_loss_2x = MSE_loss(sr2x_, lr2x_) #+ 0.1*BCE_loss(1-sr2x_, 1-lr2x_)

                    #### Perceptual Loss ####
                    real_feature = featuremapping(lr2x_)
                    fake_feature = featuremapping(sr2x_)
                    vgg_loss_2x = MSE_loss(fake_feature, real_feature.detach())

                    #### Adversarial Loss ####
                    advs_loss_2x = BCE_loss(discnet(sr2x_), real_label) if epoch + 1 > self.pretrain_epochs else 0
                    # advs_loss_2x = 0

                    #============== loss backward ===============#
                    total_loss_2x = a1 * edge_loss_2x + a2 * advs_loss_2x + \
                        a3 * content_loss_2x + (1.0 - a3) * vgg_loss_2x

                    # total_loss_2x = 1.0 * content_loss_2x + 0.25 * vgg_loss_2x

                    total_loss_2x.backward()
                    srresnet2x1_optimizer.step()

                #============ calculate scores ==============#
                # psnr_2x_score_process = batch_compare_filter(
                #     sr2x_.cpu().data, lr2x, PSNR)
                # psnr_2x_avg.add(psnr_2x_score_process)

                # ssim_2x_score_process = batch_compare_filter(
                #     sr2x_.cpu().data, lr2x, SSIM)
                # ssim_2x_avg.add(ssim_2x_score_process)

                #============ calculate 4x loss ==============#
                if is_fine_tune:
                    sr4x_ = srresnet2x2(srresnet2x1(lr4x_))

                srresnet2x2_optimizer.zero_grad()
                #### Edgenet Loss ####
                pred = edgenet(sr4x_)
                real = edgenet(hr_)

                # edge_loss_4x = 0
                edge_loss_4x = BCE_loss(pred.detach(), real.detach())
                # for i in range(6):
                #     edge_loss_4x += edge_trade_off[i] * \
                #         BCE_loss(pred[i].detach(), real[i].detach())
                # edge_loss = 0.7 * BCE2d(pred[0], real[i]) + 0.3 * BCE2d(pred[5], real[i])

                #### Content Loss ####
                content_loss_4x = MSE_loss(sr4x_, hr_) #+ 0.1*BCE_loss(1-sr4x_, 1-hr_)

                #### Perceptual Loss ####
                real_feature = featuremapping(hr_)
                fake_feature = featuremapping(sr4x_)
                vgg_loss_4x = MSE_loss(fake_feature, real_feature.detach())

                #### Adversarial Loss ####
                advs_loss_4x = BCE_loss(discnet(sr4x_), real_label) if epoch + 1 > self.pretrain_epochs else 0
                # advs_loss_4x = 0

                #============== loss backward ===============#
                total_loss_4x = a1 * edge_loss_4x + a2 * advs_loss_4x + \
                    a3 * content_loss_4x + (1.0 - a3) * vgg_loss_4x

                # total_loss_4x = 1.0 * content_loss_4x + 0.25 * vgg_loss_4x

                total_loss_4x.backward()
                srresnet2x2_optimizer.step()

                #============ calculate scores ==============#
                # psnr_4x_score_process = batch_compare_filter(
                #     sr4x_.cpu().data, hr, PSNR)
                # psnr_4x_avg.add(psnr_4x_score_process)

                # ssim_4x_score_process = batch_compare_filter(
                #     sr4x_.cpu().data, hr, SSIM)
                # ssim_4x_avg.add(ssim_4x_score_process)

                if is_fine_tune:
                    total_avg_loss.add(total_loss_4x.data.item())
                    edge_avg_loss.add(edge_loss_4x.data.item())
                else:
                    total_avg_loss.add((total_loss_2x+total_loss_4x).data.item())
                    edge_avg_loss.add((edge_loss_2x+edge_loss_4x).data.item())
                if epoch + 1 > self.pretrain_epochs:
                    disc_avg_loss.add((advs_loss_2x+advs_loss_4x).data.item())

                if (ii+1) % self.plot_iter == self.plot_iter-1:
                    res = {'edge loss': edge_avg_loss.value()[0],
                           'generate loss': total_avg_loss.value()[0],
                           'discriminate loss': disc_avg_loss.value()[0]}
                    vis.plot_many(res, 'Deblur net Loss')

                    # psnr_2x_score_origin = batch_compare_filter(
                    #     bc2x, lr2x, PSNR)
                    # psnr_4x_score_origin = batch_compare_filter(bc4x, hr, PSNR)
                    # res_psnr = {'2x_origin_psnr': psnr_2x_score_origin,
                    #             '2x_sr_psnr': psnr_2x_score_process,
                    #             '4x_origin_psnr': psnr_4x_score_origin,
                    #             '4x_sr_psnr': psnr_4x_score_process}
                    # vis.plot_many(res_psnr, 'PSNR Score')

                    # ssim_2x_score_origin = batch_compare_filter(
                    #     bc2x, lr2x, SSIM)
                    # ssim_4x_score_origin = batch_compare_filter(bc4x, hr, SSIM)
                    # res_ssim = {'2x_origin_ssim': ssim_2x_score_origin,
                    #             '2x_sr_ssim': ssim_2x_score_process,
                    #             '4x_origin_ssim': ssim_4x_score_origin,
                    #             '4x_sr_ssim': ssim_4x_score_process}
                    # vis.plot_many(res_ssim, 'SSIM Score')

                #======================= Output result of total training processing =======================#
                itcnt += 1
                # itbar.set_description("Epoch: [%2d] [%d/%d] PSNR_2x_Avg: %.6f, SSIM_2x_Avg: %.6f, PSNR_4x_Avg: %.6f, SSIM_4x_Avg: %.6f"
                #                       % ((epoch + 1), (ii + 1), len(train_data_loader),
                #                          psnr_2x_avg.value()[0], ssim_2x_avg.value()[
                #                           0],
                #                          psnr_4x_avg.value()[0], ssim_4x_avg.value()[0]))
                itbar.set_description("Epoch: [%2d] [%d/%d]"
                                      % ((epoch + 1), (ii + 1), len(train_data_loader)))

                if (ii+1) % self.plot_iter == self.plot_iter-1:
                    # test_ = deblurnet(torch.cat([y_.detach(), x_edge], 1))
                    hr_edge = edgenet(hr_)
                    sr2x_edge = edgenet(sr2x_)
                    sr4x_edge = edgenet(sr4x_)

                    vis.images(hr_edge.cpu().data, win='HR edge predict', opts=dict(
                        title='HR edge predict'))
                    vis.images(sr2x_edge.cpu().data, win='SR2X edge predict', opts=dict(
                        title='SR2X edge predict'))
                    vis.images(sr4x_edge.cpu().data, win='SR4X edge predict', opts=dict(
                        title='SR4X edge predict'))

                    vis.images(lr2x, win='LR2X image',
                               opts=dict(title='LR2X image'))
                    vis.images(lr4x, win='LR4X image',
                               opts=dict(title='LR4X image'))
                    vis.images(bc2x, win='BC2X image',
                               opts=dict(title='BC2X image'))
                    vis.images(bc4x, win='BC4X image',
                               opts=dict(title='BC4X image'))
                    vis.images(sr2x_.cpu().data, win='SR2X image',
                               opts=dict(title='SR2X image'))
                    vis.images(sr4x_.cpu().data, win='SR4X image',
                               opts=dict(title='SR4X image'))

                    vis.images(hr, win='HR image',
                               opts=dict(title='HR image'))

                t_save_dir = 'results/train_result/'+self.train_dataset
                if not os.path.exists(t_save_dir):
                    os.makedirs(t_save_dir)

                if (epoch + 1) % self.save_epochs == 0 and (ii+1) % 200 == 0:
                    self.save_model(srresnet2x1, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x1_param_batch{}_lr{}_epoch{}'.
                                    format(self.batch_size, self.lr, epoch+1))
                    self.save_model(srresnet2x2, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x2_param_batch{}_lr{}_epoch{}'.
                                    format(self.batch_size, self.lr, epoch+1))

            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(srresnet2x1, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x1_param_batch{}_lr{}_epoch{}'.
                                format(self.batch_size, self.lr, epoch+1))
                self.save_model(srresnet2x2, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x2_param_batch{}_lr{}_epoch{}'.
                                format(self.batch_size, self.lr, epoch+1))

        # Save final trained model and results
        vis.save([self.env])
        self.save_model(srresnet2x1, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x1_param_batch{}_lr{}_epoch{}'.
                        format(self.batch_size, self.lr, self.num_epochs))
        self.save_model(srresnet2x2, os.path.join(self.save_dir, 'checkpoints', 'srunitnet'), 'srnet2x2_param_batch{}_lr{}_epoch{}'.
                        format(self.batch_size, self.lr, self.num_epochs))

    def test(self, sr2x1_path=None, sr2x2_path=None):
        test_data_dir = os.path.join(self.data_dir, self.test_dataset)
        result_data_dir = os.path.join(self.save_dir, "test_results", "2x2UnitNet_SR_"+self.test_dataset)
        if not os.path.exists(result_data_dir):
            os.makedirs(result_data_dir)

        # judge whether model exists
        if not os.path.exists(sr2x1_path):
            raise Exception('sr2x1 resnet model not exists')
        if not os.path.exists(sr2x2_path):
            raise Exception('sr2x2 resnet model not exists')

        # load network params
        # srresnet2x1 = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
        #                                        norm=NORM, activation='prelu', learn_residual=True)
        # srresnet2x2 = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
        #                                        norm=NORM, activation='prelu', learn_residual=True)
        srresnet2x1 = WDSRResnetGenerator(input_nc=3, output_nc=3, n_blocks=5)
        srresnet2x2 = WDSRResnetGenerator(input_nc=3, output_nc=3, n_blocks=5)
        srresnet2x1.load_state_dict(torch.load(sr2x1_path))
        srresnet2x2.load_state_dict(torch.load(sr2x2_path))

        if USE_GPU:
            srresnet2x1.cuda()
            srresnet2x2.cuda()

        import torchnet as tnt
        from tqdm import tqdm
        from PIL import Image
        import time

        psnr_4x_avg = tnt.meter.AverageValueMeter()
        ssim_4x_avg = tnt.meter.AverageValueMeter()

        time_avg = tnt.meter.AverageValueMeter()

        srresnet2x1.eval()
        srresnet2x2.eval()

        # processing test data
        iterbar = tqdm(os.listdir(test_data_dir))
        import cv2
        import numpy as np
        for img_name in iterbar:
            try:
                img = cv2.imread(os.path.join(test_data_dir, img_name), cv2.IMREAD_COLOR)
                img = cv2.resize(img, None, None, 0.5, 0.5, interpolation=cv2.INTER_AREA)

                h, w, c = img.shape[0], img.shape[1], img.shape[2]
                w_lr4x, h_lr4x = int(
                    w // self.scale_factor), int(h // self.scale_factor)
                w_lr2x, h_lr2x = w_lr4x * 2, h_lr4x * 2
                w_hr, h_hr = w_lr4x * self.scale_factor, h_lr4x * self.scale_factor

                w_num, h_num = w // self.crop_size, h // self.crop_size
                w_num += 1 if w % self.crop_size != 0 else 0
                h_num += 1 if h % self.crop_size != 0 else 0

                res = np.zeros((h*2, w*2, c), dtype=np.uint8)
                for i in range(w_num):
                    l = i * self.crop_size
                    l_new = l * 2
                    r = min(l+self.crop_size, w)
                    r_new = w * 2 if r == w else l_new + self.crop_size * 2
                    for j in range(h_num):
                        t = j * self.crop_size
                        t_new = t * 2
                        b = min(t+self.crop_size, h)
                        b_new = h * 2 if b == h else t_new + self.crop_size * 2
                        
                        lr = img[t:b, l:r]

                        lr = Transforms.ToTensor()(lr).unsqueeze(0)
                        if USE_GPU:
                            lr = lr.cuda()

                        sr = srresnet2x1(lr).squeeze()

                        res_sr = sr.cpu().data.clamp(0, 1).numpy().transpose(1, 2, 0)*255

                        res[t_new:b_new, l_new:r_new] = res_sr
                
                cv2.imwrite(os.path.join(result_data_dir, img_name), res)
            except IOError:
                pass
            finally:
                pass


        # for img_name in iterbar:
        #     try:
        #         img = Image.open(os.path.join(test_data_dir, img_name)).convert("RGB")
        #         transform = Transforms.RandomCrop(self.crop_size)
        #         img = transform(img)

        #         w, h = img.size[0], img.size[1]
        #         w_lr4x, h_lr4x = int(
        #             w // self.scale_factor), int(h // self.scale_factor)
        #         w_lr2x, h_lr2x = w_lr4x * 2, h_lr4x * 2
        #         # w_hr, h_hr = w_lr4x * self.scale_factor, h_lr4x * self.scale_factor

        #         # transform tensor
        #         # hr = img.resize((w_hr, h_hr), Image.ANTIALIAS)
        #         # lr2x = img.resize((w_lr2x, h_lr2x), Image.ANTIALIAS)
        #         lr4x = img.resize((w_lr4x, h_lr4x), Image.ANTIALIAS)
        #         lr4x = img.resize((w_lr2x, h_lr2x), Image.ANTIALIAS)

        #         # hr_ = Transforms.ToTensor()(hr).unsqueeze(0)
        #         # lr2x_ = Transforms.ToTensor()(lr2x).unsqueeze(0)
        #         lr4x_ = Transforms.ToTensor()(lr4x).unsqueeze(0)

        #         if USE_GPU:
        #             # hr_ = hr_.cuda()
        #             # lr2x_ = lr2x_.cuda()
        #             lr4x_ = lr4x_.cuda()

        #         torch.cuda.synchronize()
        #         start = time.time()

        #         sr4x_ = srresnet2x2(srresnet2x1(lr4x_))
        #         # sr4x_ = srresnet2x1(lr4x_)

        #         torch.cuda.synchronize()
        #         end = time.time()

        #         time_avg.add(end-start)
        #     except IOError:
        #         pass
        #     finally:
        #         pass

        #     # calculate PSNR & SSIM
        #     psnr_4x_score = batch_compare_filter(
        #         sr4x_.cpu().data, hr_, PSNR)
        #     ssim_4x_score = batch_compare_filter(
        #         sr4x_.cpu().data, hr_, SSIM)
        #     psnr_4x_avg.add(psnr_4x_score)
        #     ssim_4x_avg.add(ssim_4x_score)

        #     # save image
        #     save_img(sr4x_.cpu().data, os.path.join(result_data_dir, img_name))

        print(time_avg.value()[0])
        print("final PSNR score: {}".format(psnr_4x_avg.value()[0]))
        print("final SSIM score: {}".format(ssim_4x_avg.value()[0]))

    def test_t(self, sr2x1_1_path=None, sr2x2_1_path=None, sr2x1_2_path=None, sr2x2_2_path=None):
        test_data_dir = os.path.join(self.data_dir, self.test_dataset)
        
        sr_edge_dir = os.path.join(self.save_dir, "show_results", "2x2UnitNet_Edge_SR_"+self.test_dataset)
        if not os.path.exists(sr_edge_dir):
            os.makedirs(sr_edge_dir)

        sr_none_dir = os.path.join(self.save_dir, "show_results", "2x2UnitNet_none_SR_"+self.test_dataset)
        if not os.path.exists(sr_none_dir):
            os.makedirs(sr_none_dir)
        
        bc_dir = os.path.join(self.save_dir, "show_results", "Bicubic_SR_"+self.test_dataset)
        if not os.path.exists(bc_dir):
            os.makedirs(bc_dir)
        
        hr_dir = os.path.join(self.save_dir, "show_results", "HR_"+self.test_dataset)
        if not os.path.exists(hr_dir):
            os.makedirs(hr_dir)

        lr_dir = os.path.join(self.save_dir, "show_results", "LR_"+self.test_dataset)
        if not os.path.exists(lr_dir):
            os.makedirs(lr_dir)

        # judge whether model exists
        if not os.path.exists(sr2x1_1_path):
            raise Exception('sr2x1 resnet model not exists')
        if not os.path.exists(sr2x2_1_path):
            raise Exception('sr2x2 resnet model not exists')
        if not os.path.exists(sr2x1_2_path):
            raise Exception('sr2x1 resnet model not exists')
        if not os.path.exists(sr2x2_2_path):
            raise Exception('sr2x2 resnet model not exists')

        # load network params
        srresnet2x1_edge = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
                                               norm=NORM, activation='prelu', learn_residual=True)
        srresnet2x2_edge = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
                                               norm=NORM, activation='prelu', learn_residual=True)
        srresnet2x1_none = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
                                               norm=NORM, activation='prelu', learn_residual=True)
        srresnet2x2_none = Upscale2xResnetGenerator(input_nc=3, output_nc=3, n_blocks=5,
                                               norm=NORM, activation='prelu', learn_residual=True)
        srresnet2x1_edge.load_state_dict(torch.load(sr2x1_1_path))
        srresnet2x2_edge.load_state_dict(torch.load(sr2x2_1_path))
        srresnet2x1_none.load_state_dict(torch.load(sr2x1_2_path))
        srresnet2x2_none.load_state_dict(torch.load(sr2x2_2_path))

        if USE_GPU:
            srresnet2x1_edge.cuda()
            srresnet2x2_edge.cuda()
            srresnet2x1_none.cuda()
            srresnet2x2_none.cuda()

        import torchnet as tnt
        from tqdm import tqdm
        from PIL import Image

        psnr_edge_4x_avg = tnt.meter.AverageValueMeter()
        ssim_edge_4x_avg = tnt.meter.AverageValueMeter()
        psnr_none_4x_avg = tnt.meter.AverageValueMeter()
        ssim_none_4x_avg = tnt.meter.AverageValueMeter()

        # srresnet2x1_edge.eval()
        # srresnet2x2_edge.eval()
        # srresnet2x1_none.eval()
        # srresnet2x2_none.eval()

        # processing test data
        iterbar = tqdm(os.listdir(test_data_dir))
        for img_name in iterbar:
            img = Image.open(os.path.join(test_data_dir, img_name)).convert("RGB")
            transform = Transforms.RandomCrop(self.crop_size)
            img = transform(img)

            w, h = img.size[0], img.size[1]
            w_lr4x, h_lr4x = int(
                w // self.scale_factor), int(h // self.scale_factor)
            w_hr, h_hr = w_lr4x * self.scale_factor, h_lr4x * self.scale_factor

            # transform tensor
            hr = img.resize((w_hr, h_hr), Image.ANTIALIAS)
            lr4x = img.resize((w_lr4x, h_lr4x), Image.ANTIALIAS)
            bc4x = lr4x.resize((w_hr, h_hr), Image.BICUBIC)

            hr_ = Transforms.ToTensor()(hr).unsqueeze(0)
            bc4x_ = Transforms.ToTensor()(bc4x).unsqueeze(0)
            lr4x_ = Transforms.ToTensor()(lr4x).unsqueeze(0)

            if USE_GPU:
                hr_ = hr_.cuda()
                lr4x_ = lr4x_.cuda()

            sr4x_edge_ = srresnet2x2_edge(srresnet2x1_edge(lr4x_))
            sr4x_none_ = srresnet2x2_none(srresnet2x1_none(lr4x_))

            # calculate PSNR & SSIM
            psnr_edge_4x_score = batch_compare_filter(
                sr4x_edge_.cpu().data, hr_, PSNR)
            ssim_edge_4x_score = batch_compare_filter(
                sr4x_edge_.cpu().data, hr_, SSIM)
            psnr_edge_4x_avg.add(psnr_edge_4x_score)
            ssim_edge_4x_avg.add(ssim_edge_4x_score)

            psnr_none_4x_score = batch_compare_filter(
                sr4x_none_.cpu().data, hr_, PSNR)
            ssim_none_4x_score = batch_compare_filter(
                sr4x_none_.cpu().data, hr_, SSIM)
            psnr_none_4x_avg.add(psnr_none_4x_score)
            ssim_none_4x_avg.add(ssim_none_4x_score)

            # save image
            save_img(sr4x_edge_.cpu().data, os.path.join(sr_edge_dir, img_name))
            save_img(sr4x_none_.cpu().data, os.path.join(sr_none_dir, img_name))
            save_img(bc4x_.cpu().data, os.path.join(bc_dir, img_name))
            save_img(hr_.cpu().data, os.path.join(hr_dir, img_name))
            save_img(lr4x_.cpu().data, os.path.join(lr_dir, img_name))

        print("final edge PSNR score: {}".format(psnr_edge_4x_avg.value()[0]))
        print("final edge SSIM score: {}".format(ssim_edge_4x_avg.value()[0]))

        print("final none PSNR score: {}".format(psnr_none_4x_avg.value()[0]))
        print("final none SSIM score: {}".format(ssim_none_4x_avg.value()[0]))

    def save_model(self, model, save_dir, model_name, mtype='pkl'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if mtype == 'pkl':
            save_path = os.path.join(save_dir, model_name+'.pkl')
            torch.save(model.state_dict(), save_path)
        elif mtype == 'pth':
            save_path = os.path.join(save_dir, model_name+'.pth')
            torch.save(model.state_dict(), save_path)
