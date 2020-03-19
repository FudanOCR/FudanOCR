# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from .utils import cat


class LevelMapper(nn.Module):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, scales, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            scales (list)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        super(LevelMapper, self).__init__()
        self.scales = scales
        self.k_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
        self.k_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def convert_to_roi_format(self, boxes):
        """
        :param boxes:
        :return: rois list(batch_idx, x, y, w, h)
        """
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch.full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            results (list[dict])
        """
        rois = self.convert_to_roi_format(boxes)

        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxes]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        levels = target_lvls.to(torch.int64) - self.k_min

        # for each level, crop feature maps in the rois of this level
        results = []
        for level, per_level_feature in enumerate(x):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            for batch_idx, ori_x, ori_y, ori_w, ori_h in rois_per_level:
                batch_idx = int(batch_idx)
                x = (int(ori_x * self.scales[level]) // 2) * 2
                y = (int(ori_y * self.scales[level]) // 2) * 2
                w = (int(ori_w * self.scales[level]) // 2) * 2
                h = (int(ori_h * self.scales[level]) // 2) * 2
                crop = per_level_feature[batch_idx:batch_idx+1, :, y:y+h, x:x+w]
                # rescale to the same level 0
                for i in range(level):
                    crop = nn.functional.interpolate(crop, scale_factor=2, mode='bilinear', align_corners=True)
                    x *= 2
                    y *= 2
                # save to results
                results.append({
                    'batch_idx': batch_idx,
                    'feature_map': crop,
                    'roi': [x, y, crop.shape[3], crop.shape[2]]
                })
        return results
