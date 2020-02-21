import torch
import torch.nn as nn
from model.detection_model.AdvancedEAST.network.resnet import resnet50


class East(nn.Module):
    def __init__(self):
        super(East, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(3072, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(640, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(320, 64, 1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(32, 1, 1)
        self.conv9 = nn.Conv2d(32, 2, 1)
        self.conv10 = nn.Conv2d(32, 4, 1)

        self.unpool1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.unpool3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, images):
        f = self.resnet(images)
        h = f[3]  # bs 2048 w/32 h/32
        g = self.unpool1(h)  # bs 2048 w/16 h/16
        c = self.conv1(torch.cat((g, f[2]), 1))
        c = self.bn1(c)
        c = self.relu1(c)

        h = self.conv2(c)  # bs 128 w/16 h/16
        h = self.bn2(h)
        h = self.relu2(h)
        g = self.unpool2(h)  # bs 128 w/8 h/8
        c = self.conv3(torch.cat((g, f[1]), 1))
        c = self.bn3(c)
        c = self.relu3(c)

        h = self.conv4(c)  # bs 64 w/8 h/8
        h = self.bn4(h)
        h = self.relu4(h)
        g = self.unpool3(h)  # bs 64 w/4 h/4
        c = self.conv5(torch.cat((g, f[0]), 1))
        c = self.bn5(c)
        c = self.relu5(c)

        h = self.conv6(c)  # bs 32 w/4 h/4
        h = self.bn6(h)
        h = self.relu6(h)
        g = self.conv7(h)  # bs 32 w/4 h/4
        g = self.bn7(g)
        g = self.relu7(g)

        inside_score = self.conv8(g)  # bs 1 w/4 h/4
        side_v_code = self.conv9(g)
        side_v_coord = self.conv10(g)

        east_detect = torch.cat((inside_score, side_v_code, side_v_coord), 1)
        # transpose for loss calculation
        return east_detect.transpose(1, 2).transpose(2, 3)
