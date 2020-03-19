# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/home/shf/fudan_ocr_system/datasets/"
    DATASETS = {
        "ArT_train": (
            "ICDAR19/train_images",
            "ICDAR19/train_labels.json",
        ),
        "ArT_test": (
            "ICDAR19/test_images",
            "ICDAR19/train_labels.json",
        ),
        "ArT_demo": (
            "ICDAR19/test_images",
            "ICDAR19/train_labels.json",
        ),
        # "ArT_totaltext": (
        #     "totaltext/images",
        #     "totaltext/new_train_labels.json"
        # ),
        "LSVT_train": (
            "LSVT_full_train/train_images",
            "LSVT_full_train/new_train_labels.json",
        ),
        "LSVT_train_copy": (
            "LSVT_full_train/train_images",
            "LSVT_full_train/new_train_labels.json",
        ),
        "LSVT_test": (
            "LSVT_full_train/test_images",
            "LSVT_full_train/new_train_labels.json",
        ),
        "LSVT_demo": (
            "LSVT_full_train/demo_images",
            "LSVT_full_train/new_train_labels.json",
        ),
        "LSVT_weak_0": (
            "LSVT_weak/images_0",
            "LSVT_weak/new_train_weak_labels_0.json",
        ),
        "LSVT_weak_1": (
            "LSVT_weak/images_1",
            "LSVT_weak/new_train_weak_labels_1.json",
        ),
        "LSVT_weak_3": (
            "LSVT_weak/images_3",
            "LSVT_weak/new_train_weak_labels_3.json",
        ),
        "LSVT_weak_4": (
            "LSVT_weak/images_4",
            "LSVT_weak/new_train_weak_labels_4.json",
        ),
        "LSVT_weak_5": (
            "LSVT_weak/images_5",
            "LSVT_weak/new_train_weak_labels_5.json",
        ),
        "LSVT_weak_6": (
            "LSVT_weak/images_6",
            "LSVT_weak/new_train_weak_labels_6.json",
        ),
        "LSVT_weak_7": (
            "LSVT_weak/images_7",
            "LSVT_weak/new_train_weak_labels_7.json",
        ),
        "LSVT_weak_8": (
            "LSVT_weak/images_8",
            "LSVT_weak/new_train_weak_labels_8.json",
        ),
        "LSVT_weak_9": (
            "LSVT_weak/images_9",
            "LSVT_weak/new_train_weak_labels.json",
        ),
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        elif "ArT" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="ArTDataset",
                args=args,
            )
        elif "LSVT" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="LSVTDataset",
                args=args,
            )


        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "http://pgms79tvn.bkt.clouddn.com"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "R-50.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
