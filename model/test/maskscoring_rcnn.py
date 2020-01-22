# -*- coding: utf-8 -*-

def test_maskscoring_rcnn(config_file):

    import sys
    sys.path.append('./detection_model/maskscoring_rcnn')

    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import torch
    #from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data import make_data_loader
    from maskrcnn_benchmark.engine.inference import inference
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    from maskrcnn_benchmark.utils.collect_env import collect_env_info
    from maskrcnn_benchmark.utils.comm import synchronize, get_rank
    from maskrcnn_benchmark.utils.logger import setup_logger
    from maskrcnn_benchmark.utils.miscellaneous import mkdir

    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.deprecated.init_process_group(
            backend="nccl", init_method="env://"
        )

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(opt)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(opt)
    model.to(opt.MODEL.DEVICE)

    output_dir = opt.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(opt, model, save_dir=output_dir)
    _ = checkpointer.load(opt.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if opt.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(opt.DATASETS.TEST)
    if opt.OUTPUT_DIR:
        dataset_names = opt.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(opt.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(opt, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            iou_types=iou_types,
            box_only=opt.MODEL.RPN_ONLY,
            device=opt.MODEL.DEVICE,
            expected_results=opt.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=opt.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            maskiou_on=opt.MODEL.MASKIOU_ON
        )
        synchronize()
