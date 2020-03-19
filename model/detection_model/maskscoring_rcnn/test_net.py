# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import json
import tempfile
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.engine.extra_utils import coco_results_to_contest, mask_nms
from maskrcnn_benchmark.utils.imports import import_file


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_ms_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        _, coco_results, _ = inference(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            maskiou_on=cfg.MODEL.MASKIOU_ON
        )
        synchronize()

    #############################
    # post-processing
    #############################
    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog

    output_results, bbox_results = coco_results_to_contest(coco_results)
    if cfg.TEST.VIZ:
        gt_path = os.path.join(DatasetCatalog.DATA_DIR,
                               DatasetCatalog.DATASETS[cfg.DATASETS.TEST[0]][1])
        with open(gt_path, 'r') as f:
            gt_results = json.load(f)

    # mask_nms
    mmi_thresh = 0.3
    conf_thresh = 0.5      # 0.4
    for idx, (key, result) in enumerate(output_results.items()):
        print("[ {} ]/[ {} ]".format(idx+1, len(output_results)))

        output_results[key] = mask_nms(result, result[0]['size'], mmi_thres=mmi_thresh, conf_thres=conf_thresh)
        # viz
        if cfg.TEST.VIZ:
            import cv2

            if not os.path.exists(cfg.VIS_DIR):
                os.mkdir(cfg.VIS_DIR)
            img_dir = os.path.join(DatasetCatalog.DATA_DIR,
                                   DatasetCatalog.DATASETS[cfg.DATASETS.TEST[0]][0])

            img = cv2.imread(os.path.join(img_dir, key.replace('res', 'gt')+'.jpg'))
            gt_img = img.copy()
            for rect in bbox_results[key]:
                if rect['confidence'] > conf_thresh:
                    pred_pts = rect['points']
                    img = cv2.polylines(img, [np.array(pred_pts).astype(np.int32)], True, (0, 255, 0), 3)

            for poly in output_results[key]:
                pred_pts = poly['points']
                img = cv2.polylines(img, [np.array(pred_pts).astype(np.int32)], True, (0, 0, 255), 2)

            for rect in bbox_results[key]:
                if rect['confidence'] > conf_thresh:
                    pred_pts = rect['points']
                    img = cv2.putText(img, '{:.4f}'.format(rect['confidence']), (pred_pts[0][0], pred_pts[0][1]),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    img = cv2.putText(img, '{:.4f}'.format(rect['confidence']), (pred_pts[0][0], pred_pts[0][1]),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            for gt_poly in gt_results[key.replace('res', 'gt')]['polygons']:
                gt_pts = gt_poly['points']
                if gt_poly['illegibility']:
                    gt_img = cv2.polylines(gt_img, [np.array(gt_pts).astype(np.int32)], True, (0, 255, 0), 2)
                else:
                    gt_img = cv2.polylines(gt_img, [np.array(gt_pts).astype(np.int32)], True, (0, 0, 255), 2)

            img_show = np.concatenate([img, gt_img], axis=1)
            cv2.imwrite(os.path.join(cfg.VIS_DIR, key.replace('res', 'gt')+'.jpg'), img_show)

    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "result.json")
            bbox_file_path = os.path.join(output_folder, "bbox_result.json")
        with open(file_path, "w") as json_f:
            json.dump(output_results, json_f)
        with open(bbox_file_path, "w") as json_ff:
            json.dump(bbox_results, json_ff)


if __name__ == "__main__":
    main()
