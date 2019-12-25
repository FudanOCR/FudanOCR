import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..utils import ConvModule
from ..registry import NECKS
from .denseASPP import denseASPP

@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 normalize=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.with_bias = normalize is None

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1]
                               if i == 0 else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    normalize=normalize,
                    bias=self.with_bias,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # self.downsamples = nn.Sequential()
        # for i in range(4):
        #     self.downsamples.append(f'downsample{i}', nn.MaxPool2d(3, 2, padding=1))

        self.bottom_up_blocks = nn.Sequential()
        self.latent_blocks = nn.Sequential()
        for i in range(4):
            self.bottom_up_blocks.add_module(f'bottom_up{i}', conv_block(out_channels, out_channels, 3, 2, 1))
            latent_block_module = nn.Sequential(conv_block(out_channels, out_channels // 2, 1, 1),
                                                conv_block(out_channels // 2, out_channels // 2, 3, 1),
                                                conv_block(out_channels // 2, out_channels, 1, 1))
            # latent_block_module.add_module(conv_block(in_channels, out_channels // 2, 1, 1))
            # latent_block_module.add_module(conv_block(out_channels // 2, out_channels // 2, 3, 1))
            # latent_block_module.add_module(conv_block(out_channels // 2, out_channels, 1, 1))
            self.latent_blocks.add_module(f'latent_block{i}', latent_block_module)

        # self.denseASPP_blocks = nn.Sequential()
        # for i in range(5):
        #     self.denseASPP_blocks.add_module(f'aspp_{i}', denseASPP())

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        print("Finished init weights")

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # lowest_feature = inputs[0]
        # middle_feature = lowest_feature
        # extra_feature = []
        # for downsample in self.downsamples:
        #     middle_feature = downsample(middle_feature)
        #     extra_feature.append(middle_feature)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                orig = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](orig))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    # BUG: we should add relu before each extra conv
                    outs.append(self.fpn_convs[i](outs[-1]))


        # for i, out in enumerate(outs[1:], 1):
        #     outs[i] = out + extra_feature[i]

        # for i, aspp in enumerate(self.denseASPP_blocks):
        #     outs[i] = aspp(outs[i])

        #
        # outputs = []
        # last_latent = outs[0]
        # outputs.append(last_latent[0])
        # for feature, bottom_up_block, latent_block in zip(outs[1:], self.bottom_up_blocks, self.latent_blocks):
        #     latent_feature = bottom_up_block(last_latent)
        #     mid_feature =  feature + latent_feature
        #     last_latent = latent_block(mid_feature)
        #     outputs.append(last_latent)
        # #
        # for i, output in enumerate(outs[1:], 1):
        #     outputs[i] = output + extra_feature[i]



        return tuple(outs)
        # return tuple(outs)

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, use_bn=True, use_relu=True):
    layer = nn.Sequential()
    layer.add_module('conv', nn.Conv2d(in_channels,
                                       out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
    if use_bn:
        layer.add_module('bn', nn.BatchNorm2d(out_channels))
    if use_relu:
        layer.add_module('relu', nn.ReLU())
    return layer
