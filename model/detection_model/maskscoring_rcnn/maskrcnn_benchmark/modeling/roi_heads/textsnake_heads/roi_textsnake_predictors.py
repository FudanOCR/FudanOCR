# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_TEXTSNAKE_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class TextsnakeC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(TextsnakeC4Predictor, self).__init__()
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_TEXTSNAKE_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask_tr = Conv2d(num_inputs, dim_reduced, 3, stride=1, padding=1)
        self.mask_fcn_logits_tr = Conv2d(dim_reduced, self.num_classes, 1, 1, 0)

        self.conv5_mask_tcl = Conv2d(num_inputs, dim_reduced, 3, stride=1, padding=1)
        self.mask_fcn_logits_tcl = Conv2d(dim_reduced, self.num_classes, 1, 1, 0)

        self.conv5_mask_geo = Conv2d(num_inputs, dim_reduced, 3, stride=1, padding=1)
        self.mask_fcn_logits_geo = Conv2d(dim_reduced, 1, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, crop_feature_list, features):
        batch_size = features[0].shape[0]
        output_channels = self.num_classes * 2 + 1
        H = features[0].shape[2]    # 1/4 of the input height
        W = features[0].shape[3]    # 1/4 of the input width
        device, dtype = crop_feature_list[0]['feature_map'].device, crop_feature_list[0]['feature_map'].dtype
        mask_logits = torch.zeros((batch_size, output_channels, H, W), dtype=dtype, device=device)
        for crop_feature in crop_feature_list:
            x = crop_feature['feature_map']  # (1, C, H ,W)
            batch_idx = crop_feature['batch_idx']
            roi_x, roi_y, roi_w, roi_h = crop_feature['roi']

            # conv
            tr = self.mask_fcn_logits_tr(F.relu(self.conv5_mask_tr(x)))
            tcl = self.mask_fcn_logits_tcl(F.relu(self.conv5_mask_tcl(x)))
            geo = self.mask_fcn_logits_geo(F.relu(self.conv5_mask_geo(x)))

            # accumulate
            mask_logits[batch_idx:batch_idx+1, 0:2, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] += tr
            mask_logits[batch_idx:batch_idx+1, 2:4, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] += tcl
            mask_logits[batch_idx:batch_idx+1, 4:, roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] += geo

        return mask_logits


_ROI_TEXTSNAKE_PREDICTOR = {
    "MaskRCNNC4Predictor": MaskRCNNC4Predictor,
    "TextsnakeC4Predictor": TextsnakeC4Predictor
}


def make_roi_textsnake_predictor(cfg):
    func = _ROI_TEXTSNAKE_PREDICTOR[cfg.MODEL.ROI_TEXTSNAKE_HEAD.PREDICTOR]
    return func(cfg)
