# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.structures.bounding_box import RBoxList


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # # debug
        # import cv2
        # from maskrcnn_benchmark.engine.extra_utils import xywha_to_xyxy
        # img = cv2.imread('/workspace/mnt/group/ocr/qiutairu/dataset/LSVT_full_train/demo_images/gt_0.jpg')
        # img_copy = img.copy()
        # for idx, proposal in enumerate(proposals):
        #     rbox_list = proposal.bbox.cpu().numpy()
        #     rbox_scores = proposal.get_field("objectness").cpu().numpy()
        #     # rbox_labels = proposal.get_field('labels').cpu().numpy()
        #     # target_list = targets[idx].bbox.cpu().numpy()
        #     for rbox_idx, rbox in enumerate(rbox_list):
        #         if rbox_scores[rbox_idx] > 0.5:
        #             xc, yc, w, h, a = rbox
        #             pts = xywha_to_xyxy((xc, yc, w, h, a))
        #             cv2.polylines(img, [pts], True, (0, 255, 0), 3)
        #
        #     # for gt_box in target_list:
        #     #     xc, yc, w, h, a = gt_box
        #     #     pts = xywha_to_xyxy((xc, yc, w, h, a))
        #     #     cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        #
        # cv2.imwrite('debug_proposals.jpg', img)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        expanded_proposals = self.expand_proposals(proposals, self.cfg)
        x1 = self.feature_extractor(features, proposals)    # used for classification
        x2 = self.feature_extractor(features, expanded_proposals)   # used for regression
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x1, x2)

        # # debug
        # results = self.post_processor((class_logits, box_regression), proposals)
        # for idx, result in enumerate(results):
        #     pre_rbox_list = result.bbox.detach().cpu().numpy()
        #     pre_rbox_scores = result.get_field('scores').detach().cpu().numpy()
        #     # target_list = targets[idx].bbox.cpu().numpy()
        #     for rbox_idx, rbox in enumerate(pre_rbox_list):
        #         if pre_rbox_scores[rbox_idx] > 0.05:
        #             xc, yc, w, h, a = rbox
        #             pts = xywha_to_xyxy((xc, yc, w, h, a))
        #             cv2.polylines(img_copy, [pts], True, (0, 255, 0), 3)
        #
        #     # for gt_box in target_list:
        #     #     xc, yc, w, h, a = gt_box
        #     #     pts = xywha_to_xyxy((xc, yc, w, h, a))
        #     #     cv2.polylines(img_copy, [pts], True, (0, 0, 255), 2)
        #
        # cv2.imwrite('debug_predicts.jpg', img_copy)
        # print(sssss)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x1, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x1,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )

    def expand_proposals(self, boxes, cfg):
        assert isinstance(boxes, (list, tuple))
        assert isinstance(boxes[0], RBoxList)
        new_boxes = []
        for boxes_per_image in boxes:
            proposals = boxes_per_image.bbox.clone()
            im_info = boxes_per_image.size
            proposals[:, 2:4] *= cfg.MODEL.RRPN.GT_BOX_MARGIN
            new_boxes_per_image = RBoxList(proposals, im_info, mode="xywha")
            new_boxes_per_image._copy_extra_fields(boxes_per_image)
            new_boxes.append(new_boxes_per_image)
        return new_boxes


def build_roi_box_head(cfg):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg)
