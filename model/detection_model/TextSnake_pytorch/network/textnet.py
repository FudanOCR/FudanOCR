import torch.nn as nn
import torch
import torch.nn.functional as F
from network.vgg import VGG16
from network.resnet import ResNet50
import json


class GCN(nn.Module):
    def __init__(self, c, out_c, k=(7, 7)):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding=(int((k[0] - 1) / 2), 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding=(0, int((k[0] - 1) / 2)))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k[1]), padding=(0, int((k[1] - 1) / 2)))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding=(int((k[1] - 1) / 2), 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)

        x = x + x_res

        return x


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, backbone='vgg'):
        super().__init__()
        self.backbone = backbone
        if backbone == 'vgg':
            self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        elif backbone == 'resnet':
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut, is_deconv=True):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        if is_deconv:
            x = self.deconv(x)
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class TextNet(nn.Module):
    def __init__(self, cfg, alphabet):
        super().__init__()

        self.backbone_name = cfg.BASE.NETWORK
        self.output_channel = cfg.output_channel
        self.bottleneck = 32

        if self.backbone_name == 'vgg':
            self.backbone = VGG16()
            self.deconv5 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.merge4 = Upsample(512 + 512, 256)
            self.merge3 = Upsample(256 + 256, 128)
            self.merge2 = Upsample(128 + 128, 64)
            self.merge1 = Upsample(64 + 64, self.output_channel)

        elif self.backbone_name == 'resnet':
            self.backbone = ResNet50()
            self.deconv5 = nn.ConvTranspose2d(self.output_channel, self.output_channel, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(self.output_channel, self.output_channel, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(self.output_channel, self.output_channel, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(self.output_channel, self.output_channel, kernel_size=4, stride=2, padding=1)
            self.deconv1 = nn.ConvTranspose2d(self.output_channel, self.output_channel, kernel_size=4, stride=2, padding=1)

            self.gcn5 = GCN(2048, self.output_channel)
            self.gcn4 = GCN(1024, self.output_channel)
            self.gcn3 = GCN(512, self.output_channel)
            self.gcn2 = GCN(256, self.output_channel)

            self.br5 = BR(self.output_channel)
            self.br4_1 = BR(self.output_channel)
            self.br4_2 = BR(self.output_channel)
            self.br3_1 = BR(self.output_channel)
            self.br3_2 = BR(self.output_channel)
            self.br2_1 = BR(self.output_channel)
            self.br2_2 = BR(self.output_channel)
            self.br1 = BR(self.output_channel)
            self.br0 = BR(self.output_channel)

        elif self.backbone_name == 'resnet_gcn':
            self.backbone = ResNet50()
            self.deconv5 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv1_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)   # tr
            self.deconv1_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)   # tcl

            self.gcn5 = GCN(2048, self.bottleneck)
            self.gcn4 = GCN(1024, self.bottleneck)
            self.gcn3 = GCN(512, self.bottleneck)
            self.gcn2 = GCN(256, self.bottleneck)
            self.gcn1_1 = GCN(self.bottleneck, 2)   # tr
            self.gcn1_2 = GCN(self.bottleneck, 2)   # tcl

            self.br5 = BR(self.bottleneck)
            self.br4_1 = BR(self.bottleneck)
            self.br4_2 = BR(self.bottleneck)
            self.br3_1 = BR(self.bottleneck)
            self.br3_2 = BR(self.bottleneck)
            self.br2_1 = BR(self.bottleneck)
            self.br2_2 = BR(self.bottleneck)

            self.br1_1 = BR(2)        # tr
            self.br1_2 = BR(2)        # tcl
            self.br0_1 = BR(2)        # tr
            self.br0_2 = BR(2)        # tcl

            self.conv1 = nn.Sequential(
                nn.Conv2d(self.bottleneck, self.bottleneck, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.bottleneck, 3, kernel_size=1, stride=1, padding=0)        # geo(sin, cos, radii)
            )

        elif self.backbone_name == 'resnet_gcn_new':
            self.backbone = ResNet50()
            self.deconv5 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv1_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)   # tr
            self.deconv1_2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)   # tcl

            self.gcn5 = GCN(2048, self.bottleneck)
            self.gcn4 = GCN(1024, self.bottleneck)
            self.gcn3 = GCN(512, self.bottleneck)
            self.gcn2 = GCN(256, self.bottleneck)

            self.br5 = BR(self.bottleneck)
            self.br4_1 = BR(self.bottleneck)
            self.br4_2 = BR(self.bottleneck)
            self.br3_1 = BR(self.bottleneck)
            self.br3_2 = BR(self.bottleneck)
            self.br2_1 = BR(self.bottleneck)
            self.br2_2 = BR(self.bottleneck)

            self.br1_1 = BR(2)        # tr
            self.br1_2 = BR(2)        # tcl
            self.br0_1 = BR(2)        # tr
            self.br0_2 = BR(2)        # tcl

            self.conv_tr = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)

            self.conv_geo = nn.Sequential(
                nn.Conv2d(self.bottleneck, self.bottleneck, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.bottleneck, 3, kernel_size=1, stride=1, padding=0)        # geo(sin, cos, radii)
            )

        elif self.backbone_name == 'resnet_gcn_ms':
            self.backbone = ResNet50()
            self.deconv5 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2_1 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2_2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2_3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv1_1 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv1_2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)

            self.gcn5 = GCN(2048, self.bottleneck)
            self.gcn4 = GCN(1024, self.bottleneck)
            self.gcn3 = GCN(512, self.bottleneck)
            self.gcn2 = GCN(256, self.bottleneck)

            self.br5 = BR(self.bottleneck)
            self.br4_1 = BR(self.bottleneck)
            self.br4_2 = BR(self.bottleneck)
            self.br3_1 = BR(self.bottleneck)
            self.br3_2 = BR(self.bottleneck)
            self.br2_1 = BR(self.bottleneck)
            self.br2_2 = BR(self.bottleneck)
            self.br1_1 = BR(self.bottleneck)
            self.br1_2 = BR(self.bottleneck)
            self.br0_1 = BR(self.bottleneck)
            self.br0_2 = BR(self.bottleneck)

            self.conv_tr_128 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl_128 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tr_256 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl_256 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tr_512 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl_512 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)

            self.conv_geo = nn.Sequential(
                nn.Conv2d(self.bottleneck, self.bottleneck, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.bottleneck, 3, kernel_size=1, stride=1, padding=0)        # geo(sin, cos, radii)
            )

        elif self.backbone_name == 'resnet_gcn_ms2':
            self.backbone = ResNet50()
            self.deconv5 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            self.deconv2_2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv2_3 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)
            self.deconv1_1 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            self.deconv1_2 = nn.ConvTranspose2d(self.bottleneck, self.bottleneck, kernel_size=4, stride=2, padding=1)

            self.gcn5 = GCN(2048, self.bottleneck)
            self.gcn4 = GCN(1024, self.bottleneck)
            self.gcn3 = GCN(512, self.bottleneck)
            self.gcn2 = GCN(256, self.bottleneck)

            self.br5 = BR(self.bottleneck)
            self.br4_1 = BR(self.bottleneck)
            self.br4_2 = BR(self.bottleneck)
            self.br3_1 = BR(self.bottleneck)
            self.br3_2 = BR(self.bottleneck)
            self.br2_1 = BR(self.bottleneck)
            self.br2_2 = BR(self.bottleneck)

            self.br1_1 = BR(2)
            self.br1_2 = BR(self.bottleneck)
            self.br0_1 = BR(2)
            self.br0_2 = BR(self.bottleneck)

            self.conv_tcl_128 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl_256 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)
            self.conv_tcl_512 = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)

            self.conv_tr = nn.Conv2d(self.bottleneck, 2, kernel_size=1, stride=1, padding=0)

            self.conv_geo = nn.Sequential(
                nn.Conv2d(self.bottleneck, self.bottleneck, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.bottleneck, 3, kernel_size=1, stride=1, padding=0)        # geo(sin, cos, radii)
            )

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)

        if self.backbone_name == 'vgg':
            up5 = self.deconv5(C5)
            up5 = F.relu(up5)

            up4 = self.merge4(C4, up5)
            up4 = F.relu(up4)

            up3 = self.merge3(C3, up4)
            up3 = F.relu(up3)

            up2 = self.merge2(C2, up3)
            up2 = F.relu(up2)

            up1 = self.merge1(C1, up2)

        elif self.backbone_name == 'resnet':
            up5 = self.deconv5(self.br5(self.gcn5(C5)))
            up4 = self.deconv4(self.br4_2(up5 + self.br4_1(self.gcn4(C4))))
            up3 = self.deconv3(self.br3_2(up4 + self.br3_1(self.gcn3(C3))))
            up2 = self.deconv2(self.br2_2(up3 + self.br2_1(self.gcn2(C2))))
            up1 = self.br0(self.deconv1(self.br1(up2)))

        elif self.backbone_name == 'resnet_gcn':
            up5 = self.deconv5(self.br5(self.gcn5(C5)))
            up4 = self.deconv4(self.br4_2(up5 + self.br4_1(self.gcn4(C4))))
            up3 = self.deconv3(self.br3_2(up4 + self.br3_1(self.gcn3(C3))))
            up2 = self.deconv2(self.br2_2(up3 + self.br2_1(self.gcn2(C2))))

            tr_pred_map = self.br0_1(self.deconv1_1(self.br1_1(self.gcn1_1(up2))))
            tcl_pred_map = self.br0_2(self.deconv1_2(self.br1_2(self.gcn1_2(up2))))
            geo_pred_map = F.interpolate(self.conv1(up2), scale_factor=2, mode='nearest')

            up1 = torch.cat((tr_pred_map, tcl_pred_map, geo_pred_map), dim=1)

        elif self.backbone_name == 'resnet_gcn_new':
            up5 = self.deconv5(self.br5(self.gcn5(C5)))
            up4 = self.deconv4(self.br4_2(up5 + self.br4_1(self.gcn4(C4))))
            up3 = self.deconv3(self.br3_2(up4 + self.br3_1(self.gcn3(C3))))
            up2 = self.deconv2(self.br2_2(up3 + self.br2_1(self.gcn2(C2))))

            tr_pred_map = self.br0_1(self.deconv1_1(self.br1_1(self.conv_tr(up2))))
            tcl_pred_map = self.br0_2(self.deconv1_2(self.br1_2(self.conv_tcl(up2))))
            geo_pred_map = F.interpolate(self.conv_geo(up2), scale_factor=2, mode='nearest')

            up1 = torch.cat((tr_pred_map, tcl_pred_map, geo_pred_map), dim=1)

        elif self.backbone_name == 'resnet_gcn_ms':
            up5 = self.deconv5(self.br5(self.gcn5(C5)))
            up4 = self.deconv4(self.br4_2(up5 + self.br4_1(self.gcn4(C4))))
            up3 = self.deconv3(self.br3_2(up4 + self.br3_1(self.gcn3(C3))))

            # 128*128
            feature_map_128 = self.br2_2(up3 + self.br2_1(self.gcn2(C2)))
            tr_pred_128 = self.conv_tr_128(feature_map_128)    # N * 2 * 128 * 128
            tcl_pred_128 = self.conv_tcl_128(feature_map_128)  # N * 2 * 128 * 128

            # 256*256
            tr_feature_map_256 = self.br1_1(self.deconv2_1(feature_map_128 * torch.exp(tr_pred_128[:, 1:2].sigmoid())))
            tr_pred_256 = self.conv_tr_256(tr_feature_map_256)
            tcl_feature_map_256 = self.br1_2(self.deconv2_2(feature_map_128 * torch.exp(tcl_pred_128[:, 1:2].sigmoid())))
            tcl_pred_256 = self.conv_tcl_256(tcl_feature_map_256)

            # 512*512
            tr_feature_map_512 = self.br0_1(self.deconv1_1(tr_feature_map_256 * torch.exp(tr_pred_256[:, 1:2].sigmoid())))
            tr_pred_map = self.conv_tr_512(tr_feature_map_512)
            tcl_feature_map_512 = self.br0_2(self.deconv1_2(tcl_feature_map_256 * torch.exp(tcl_pred_256[:, 1:2].sigmoid())))
            tcl_pred_map = self.conv_tcl_512(tcl_feature_map_512)

            geo_pred_map = F.interpolate(self.conv_geo(self.deconv2_3(feature_map_128)), scale_factor=2, mode='nearest')

            up1 = torch.cat((tr_pred_map, tcl_pred_map, geo_pred_map), dim=1)

        elif self.backbone_name == 'resnet_gcn_ms2':
            up5 = self.deconv5(self.br5(self.gcn5(C5)))
            up4 = self.deconv4(self.br4_2(up5 + self.br4_1(self.gcn4(C4))))
            up3 = self.deconv3(self.br3_2(up4 + self.br3_1(self.gcn3(C3))))

            # 128*128
            feature_map_128 = self.br2_2(up3 + self.br2_1(self.gcn2(C2)))
            tcl_pred_128 = self.conv_tcl_128(feature_map_128)  # N * 2 * 128 * 128

            # 256*256
            tcl_feature_map_256 = self.br1_2(self.deconv2_2(feature_map_128 * torch.exp(tcl_pred_128.softmax(dim=1)[:, 1:2])))
            tcl_pred_256 = self.conv_tcl_256(tcl_feature_map_256)

            # 512*512
            tcl_feature_map_512 = self.br0_2(self.deconv1_2(tcl_feature_map_256 * torch.exp(tcl_pred_256.softmax(dim=1)[:, 1:2])))
            tcl_pred_map = self.conv_tcl_512(tcl_feature_map_512)

            tr_pred_map = self.br0_1(self.deconv1_1(self.br1_1(self.deconv2_1(self.conv_tr(feature_map_128)))))
            geo_pred_map = F.interpolate(self.conv_geo(self.deconv2_3(feature_map_128)), scale_factor=2, mode='nearest')

            up1 = torch.cat((tr_pred_map, tcl_pred_map, geo_pred_map), dim=1)

        return up1


