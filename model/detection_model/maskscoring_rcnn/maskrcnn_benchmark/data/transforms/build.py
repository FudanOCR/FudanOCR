# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        rotate_prob = 1.0
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        rotate_prob = -1

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    _aug_list = {
        "MaskRRPN": T.Compose(
            [
                T.Resize(min_size, max_size),
                T.RandomRotation(prob=-1, r_range=cfg.INPUT.ROTATION_RANGE, fixed_angle=-1, gt_margin=cfg.MODEL.RRPN.GT_BOX_MARGIN),
                T.ToTensor(),
                # T.MixUp(mix_ratio=0.1),
                normalize_transform,
            ]
        ),
        "GeneralizedRCNN": T.Compose(
            [
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    }

    return _aug_list[cfg.MODEL.META_ARCHITECTURE]
