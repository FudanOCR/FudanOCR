import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transforms

from model.base import BaseModel
from model.functions.normalize import weights_init_normal


class OCRSRModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_network(self, network_name, network_type, parameters):
        if network_type == 'WDSRResNet':
            network = WDSRResnetGenerator(**parameters)
        elif network_type == 'CommonResNet':
            network = Upscale2xResnetGenerator(**parameters)
        else:
            raise ValueError('Invalid network type {}!'.format(network_type))
       
        network.apply(weights_init_normal)
        return network

    def init_train_mode(self):
        self.build_output = self.build_train_output
        self.current_mode = 'train'
    
    def init_eval_mode(self):
        self.build_output = self.build_eval_output
        self.current_mode = 'eval'
    
    def __calculate_batch_groups(self, shape):
        h, w = shape
        batch_size = self.mode['batch_size']
        crop_size = self.mode['crop_size']

        horizontal_patch_num = w // crop_size
        vertical_patch_num = h // crop_size
        is_horizontal_left = True if w % crop_size != 0 else False
        is_vertical_left = True if h % crop_size != 0 else False

        batch_groups = []
        single_batch = []
        # crop fit patches
        for i in  range(vertical_patch_num):
            for j in range(horizontal_patch_num):
                t, b = i * crop_size, (i + 1) * crop_size
                l, r = j * crop_size, (j + 1) * crop_size

                single_batch.append((t, b, l, r))

                if (i*horizontal_patch_num + j + 1) % batch_size == 0:
                    batch_groups.append(single_batch)
                    single_batch = []
        
        if len(single_batch) != 0:
            batch_groups.append(single_batch)

        # crop horizontal left patches
        if is_horizontal_left:
            single_batch = []
            l, r = horizontal_patch_num * crop_size, w
            for i in range(vertical_patch_num):
                t, b = i * crop_size, (i + 1) * crop_size
                single_batch.append((t, b, l, r))

                if (i + 1) % batch_size == 0:
                    batch_groups.append(single_batch)
                    single_batch = []
            if len(single_batch) != 0:
                batch_groups.append(single_batch)


        # crop horizontal left patches
        if is_vertical_left:
            single_batch = []
            t, b = vertical_patch_num * crop_size, h
            for i in range(horizontal_patch_num):
                l, r = i * crop_size, (i + 1) * crop_size
                single_batch.append((t, b, l, r))

                if (i + 1) % batch_size == 0:
                    batch_groups.append(single_batch)
                    single_batch = []
            if len(single_batch) != 0:
                batch_groups.append(single_batch)
       
        # crop bottom right cornor
        if is_horizontal_left and is_vertical_left:
            single_batch = []
            t, b = horizontal_patch_num * crop_size, h
            l, r = horizontal_patch_num * crop_size, w
            single_batch.append((t, b, l, r))
            batch_groups.append(single_batch)

        return batch_groups

    def __call__(self, **kwargs):
        # get parameters
        image_list = kwargs.get('images')

        if image_list is None:
            raise ValueError('Invalid image input!')

        result_images = []
        for image in image_list:
            pred = self.run(image)
            result_images.append(pred)

        return result_images
    
    def run(self, image):
        if len(image.shape) != 3:
            raise ValueError('Input Image should be RGB 3-channel.')
        h, w, c = image.shape

        crop_size = self.mode['crop_size']
        scale_factor = self.mode['scale_factor']

        result_image = np.zeros((h*scale_factor, w*scale_factor, c), dtype=np.uint8)

        batch_groups = self.__calculate_batch_groups((h, w))
        for batch_group in batch_groups:
            # build input tensor
            ipt = []
            for patch_position in batch_group:
                t, b, l, r = patch_position
                lr = Transforms.ToTensor()(image[t:b, l:r])
                ipt.append(lr)
            
            ipt_ = torch.stack(ipt)
            if self.use_gpu:
                ipt_ = ipt_.cuda()
            
            pred_ =  self.networks['srunitnet2x1'](ipt_)
            if self.mode['scale_factor'] == 4:
                pred_ = self.networks['srunitnet2x2'](pred_)
            sr_results = pred_.cpu().data.clamp(0, 1).numpy() * 255

            for patch_idx in range(len(sr_results)):
                t, b, l, r = batch_group[patch_idx]
                sr_result = sr_results[patch_idx].transpose(1, 2, 0)

                t_new = t * scale_factor
                b_new = h * scale_factor if b == h else t_new + crop_size * scale_factor
                l_new = l * scale_factor
                r_new = w * scale_factor if r == w else l_new + crop_size * scale_factor
                result_image[t_new:b_new, l_new:r_new] = sr_result

        return result_image
    
    
#=====================================================================#
#============================== networks ==============================#
#=====================================================================#
class WDSRResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, n_downsampling=2, scale_factor=2, learn_residual=True, use_parallel=True,
                 padding_type='reflect', normalization='weight', activation='prelu', upsample='ps', resblock_type='b',
                 rgb_mean=[0.4488, 0.4371, 0.4040]):
        assert(n_blocks >= 0)

        from model.networks.wdsr_blocks import ConvBlock, UpsampleBlock, ResidualBlock_A, ResidualBlock_B

        super(WDSRResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True if normalization == 'instance' else False

        self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(rgb_mean)).view([1, 3, 1, 1])
        if torch.cuda.is_available():
            self.rgb_mean = self.rgb_mean.cuda()

        ResidualBlock = locals()['ResidualBlock_'+resblock_type.upper()]
        resnet_param_dict = {   'num_filters': ngf,
                                                        'bias': use_bias,
                                                        'padding_type': padding_type,
                                                        'normalization': normalization,
                                                        'activation': activation,
                                                        'res_scale': 1,
                                                        'expand': 3
        }
        if resblock_type == 'b':
            resnet_param_dict['linear'] = 0.8

        # head block
        self.head = ConvBlock(input_nc, ngf, 3, 1, 1, bias=use_bias,
                              padding_type=padding_type, normalization=normalization, activation=None)

        # residual body
        body = []
        for i in range(n_blocks):
            body.append(ResidualBlock(**resnet_param_dict))
        self.body = nn.Sequential(*body)

        # upsampling
        self.tail = UpsampleBlock(ngf, output_nc, scale_factor=scale_factor, upsample=upsample,
                                      normalization=normalization, activation=None)

        # skip connection
        self.skip = UpsampleBlock(input_nc, output_nc, kernel_size=5, scale_factor=scale_factor, upsample=upsample,
                                  normalization=normalization, activation=None)

    def forward(self, x):
        x = (x - self.rgb_mean)
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if self.learn_residual:
            x = x + s
        x = x + self.rgb_mean
        return torch.sigmoid(x)


class Upscale2xResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='prelu', upsample='ps',
                 padding_type='reflect', use_dropout=False, learn_residual=False):
        assert(n_blocks >= 0)

        from model.networks.baseblocks import ConvBlock, ResidualBlock, Upsample2xBlock
        
        super(Upscale2xResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        use_bias = True if norm == 'instance' else False
        self.conv1 = ConvBlock(input_nc, ngf, 9, 1, 4, bias=use_bias,
                               norm=norm, activation=activation)

        # 2x size network blocks
        resblocks = []
        for i in range(n_blocks):
            resblocks += [ResidualBlock(ngf, padding_type=padding_type,
                                        norm=norm, activation=activation,
                                        bias=use_bias, dropout=use_dropout)]
        resblocks += [ConvBlock(ngf, ngf, 3, 1, 1, bias=use_bias,
                                norm=norm, activation=None)]
        self.resblocks = nn.Sequential(*resblocks)

        self.upscale = Upsample2xBlock(ngf, ngf, upsample=upsample,
                                       bias=use_bias, norm=norm, activation=activation)
        self.outconv = ConvBlock(
            ngf, output_nc, 3, 1, 1, activation=None, norm=None)

    def forward(self, x):
        x = self.conv1(x)

        out = self.resblocks(x)
        if self.learn_residual:
            out = x + out
        out = self.upscale(out)
        out = self.outconv(out)

        return F.sigmoid(out)
