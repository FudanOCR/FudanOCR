import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transforms

from model.base import BaseModel
from model.functions.normalize import weights_init_normal

class DeblurModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

    def init_network(self, network_name, network_type, parameters):
        if network_type == 'WDSRResNet':
            network = WDSRResnetGenerator(**parameters)
        elif network_type == 'CommonResNet':
            network = ResnetGenerator(**parameters)
        elif network_type == 'HED':
            from model.networks.hed import HED_1L
            network = HED_1L(**parameters)
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

        blr_ = Transforms.ToTensor()(image).unsqueeze(0)
        if self.use_gpu:
            blr_ = blr_.cuda()
        
        edge_ = self.networks['edgenet'](blr_).detach()
        x_ = blr_ + edge_
        y_ = self.networks['deblurnet'](x_)

        result_image = y_.squeeze().cpu().data.clamp(0, 1).numpy().transpose(1, 2, 0) * 255

        return result_image
    
#=====================================================================#
#============================== networks ==============================#
#=====================================================================#
class WDSRResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, n_downsampling=2, learn_residual=True, use_parallel=True,
                 padding_type='reflect', normalization=None, activation='relu', upsample='ps', resblock_type='b',
                 rgb_mean=[0.4488, 0.4371, 0.4040]):
        assert(n_blocks >= 0)
        
        from model.networks.wdsr_blocks import ConvBlock, UpsampleBlock, ResidualBlock_A, ResidualBlock_B

        super(WDSRResnetGenerator, self).__init__()
        self.learn_residual = learn_residual
        
        use_bias = True if normalization == 'instance' else False

        self.rgb_mean = torch.autograd.Variable(
            torch.FloatTensor(rgb_mean)).view([1, 3, 1, 1])
        self.rgb_mean_e = torch.autograd.Variable(
            torch.FloatTensor(rgb_mean+[0.5])).view([1, 4, 1, 1])
        if torch.cuda.is_available():
            self.rgb_mean = self.rgb_mean.cuda()
            self.rgb_mean_e = self.rgb_mean_e.cuda()

        ResidualBlock = locals()['ResidualBlock_'+resblock_type.upper()]
        resnet_param_dict = {   'bias': use_bias,
                                                        'padding_type': padding_type,
                                                        'normalization': normalization,
                                                        'activation': activation,
                                                        'res_scale': 1,
                                                        'expand': 3
        }
        if resblock_type == 'b':
            resnet_param_dict['res_scale'] = 1.2
            resnet_param_dict['expand'] = 3
            resnet_param_dict['linear'] = 0.75

        self.conv = ConvBlock(input_nc, ngf, 7, 1, 3, bias=use_bias,
                              padding_type=padding_type, normalization=normalization, activation=activation)

        # downsampling
        down = []
        for i in range(n_downsampling):
            mult = 2 ** i
            down.append(ConvBlock(ngf * mult, ngf * mult * 2, 3, 2, 1, bias=use_bias,
                                  padding_type=None, normalization=normalization, activation=activation))
        if len(down) == 0:
            down.append(ConvBlock(ngf, ngf, 3, 1, 1, bias=use_bias, padding_type=None,
                                  normalization=normalization, activation=activation))
        self.down = nn.Sequential(*down)

        # residual body
        body = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            resnet_param_dict['num_filters'] = ngf * mult
            body.append(ResidualBlock(**resnet_param_dict))
            # body.append(ResidualBlock(ngf * mult, bias=use_bias, padding_type=padding_type,
            #                           normalization=normalization, activation=activation))
        self.body = nn.Sequential(*body)

        # upsampling
        up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            up.append(UpsampleBlock(ngf * mult, int(ngf * mult // 2), bias=use_bias,
                                    upsample=upsample, normalization=normalization, activation=activation))

        up.append(ConvBlock(ngf, output_nc, 7, 1, 3, padding_type=padding_type,
                            normalization=normalization, activation=None))
        self.up = nn.Sequential(*up)

        # skip connection
        if self.learn_residual:
            self.skip = ConvBlock(ngf, output_nc, 7, 1, 3, bias=use_bias,
                                  padding_type=padding_type, normalization=normalization, activation=None)

    def forward(self, x):
        x = (x - self.rgb_mean) / 0.5
        f = self.conv(x)
        d = self.down(f)
        h = self.body(d) + d
        if self.learn_residual:
            h = h + d
        o = self.up(h)
        if self.learn_residual:
            o += self.skip(f)
        # o = torch.clamp(o, min=0, max=1.0)
        o = o * 2 + self.rgb_mean
        o = (torch.tanh(o) + 1) / 2
        return o


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', padding_type='reflect',
                use_dropout=False, learn_residual=False, use_parallel=True, rgb_mean=[0.4488, 0.4371, 0.4040]):
        assert(n_blocks >= 0)

        from model.networks.baseblocks import ConvBlock, ResidualBlock, Upsample2xBlock

        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.learn_residual = learn_residual

        self.rgb_mean = torch.autograd.Variable(
            torch.FloatTensor(rgb_mean)).view([1, 3, 1, 1])
        self.rgb_mean_e = torch.autograd.Variable(
            torch.FloatTensor(rgb_mean+[0.5])).view([1, 4, 1, 1])
        if torch.cuda.is_available():
            self.rgb_mean = self.rgb_mean.cuda()
            self.rgb_mean_e = self.rgb_mean_e.cuda()

        use_bias = True if norm == 'instance' else False        
        model = [nn.ReflectionPad2d(3),
                ConvBlock(input_nc, ngf, 7, 1, 0, bias=use_bias,\
                            norm=norm, activation='relu')]
        
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [ConvBlock(ngf * mult, ngf * mult * 2, 3, 2, 1, bias=use_bias,\
                                norm=norm, activation='relu')]
        
        mult= 2**n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult, padding_type=padding_type,
                                        norm=norm, bias=use_bias, dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model += [DeconvBlock(ngf * mult, int(ngf * mult // 2), 4, 2, 1,
            #             bias=use_bias, norm=norm, activation='relu')]
            model += [Upsample2xBlock(ngf * mult, int(ngf * mult // 2), upsample='ps',
                                    bias=use_bias, norm=norm, activation='relu')]
        
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.tanh()
                ]
                
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x - self.rgb_mean - 0.2
        out = self.model(x)
        if self.learn_residual:
            out = torch.clamp(out + x, min=-1, max=1)
        return out
