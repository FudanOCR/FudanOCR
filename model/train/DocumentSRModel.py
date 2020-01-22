# -*- coding: utf-8 -*-
def train_documentsrmodel(config_file):
    import sys
    sys.path.append('./super_resolution_model/DocumentSRModel')

    import os

    from databuilder.rvl_cdip import build_dataset
    from utils.configloader import ConfigLoader
    from models.srunitnet_2x_2x import Model

    # from models.total import Model
    # from models.srcnn import Model

    DATA_DIR = '/home/super-videt/Storage/Resource/CV/dataset'
    NEW_DIR = './datasets'

    # build_dataset(data_dir=DATA_DIR, new_dir=NEW_DIR, mode='test')

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    edgenet_path = "/home/cjy/edgenet_param_batch4_lr0.002_epoch100.pkl"
    srcnn_path = "results/checkpoints/srcnn/srcnn_param_batch4_lr0.001_epoch3.pkl"
    sr2x1_path = "/home/cjy/srnet2x1_param_batch4_lr0.0005_epoch1.pkl"
    sr2x2_path = "/home/cjy/srnet2x2_param_batch4_lr0.0005_epoch1.pkl"
    srresnet_path = "results/checkpoints/srresnet/srresnet_param_batch4_lr0.001_epoch3.pkl"
    sr2x1_edge_path = "results/checkpoints/srunitnet/srnet2x1_param_batch4_lr0.0005_epoch47.pkl"
    sr2x2_edge_path = "results/checkpoints/srunitnet/srnet2x2_param_batch4_lr0.0005_epoch47.pkl"
    # sr2x1_path = None
    # sr2x2_path = None

    # cfg = ConfigLoader('configs/default.cfg', 'document')

    # 重构cfg解析方法

    from yacs.config import CfgNode as CN
    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    cfg = read_config_file(config_file)  # 获取了yaml文件

    # cfg = ConfigLoader(config_file, 'rvl_cdip')
    # cfg = ConfigLoader('configs/default.cfg', 'test')
    model = Model(cfg)
    model.train(sr2x1_path=sr2x1_path, sr2x2_path=sr2x2_path, edgenetpath=edgenet_path, srcnn_path=srcnn_path, srresnet_path=srresnet_path,
                random_scale=False, is_fine_tune=False, rotate=False, fliplr=False, fliptb=False)
    # model.train(sr2x1_path=sr2x1_path, sr2x2_path=sr2x2_path, edgenetpath=edgenet_path,
    #             random_scale=False, is_fine_tune=True, rotate=False, fliplr=False, fliptb=False)
    # model.train(sr2x1_path=sr2x1_path, sr2x2_path=sr2x2_path, edgenetpath=edgenet_path,
    #             random_scale=False, is_fine_tune=True, rotate=False, fliplr=False, fliptb=False)
    # model.test(sr2x1_path=sr2x1_path, sr2x2_path=sr2x2_path)
    # model.test(srcnn_path=srcnn_path)

    # sr2x1_edge_path = "results/checkpoints/srunitnet/srnet2x1_param_batch4_lr0.0005_epoch47.pkl"
    # sr2x2_edge_path = "results/checkpoints/srunitnet/srnet2x2_param_batch4_lr0.0005_epoch47.pkl"
    # sr2x1_none_path = "results/checkpoints/srunitnet/srnet2x1_param_batch4_lr0.001_epoch25.pkl"
    # sr2x2_none_path = "results/checkpoints/srunitnet/srnet2x2_param_batch4_lr0.001_epoch25.pkl"

    # model.test_t(sr2x1_1_path=sr2x1_edge_path, sr2x2_1_path=sr2x2_edge_path, sr2x1_2_path=sr2x1_none_path, sr2x2_2_path=sr2x2_none_path)
