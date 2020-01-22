import os

from databuilder.rvl_cdip import build_dataset
from utils.configloader import ConfigLoader
from models.srcnn import Model
# from models.srresnet import Model

DATA_DIR = '/home/super-videt/Storage/Resource/CV/dataset'
NEW_DIR = './datasets'

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    srcnn_path = "results/checkpoints/srcnn/srcnn_param_batch4_lr0.001_epoch10.pkl"
    
    cfg = ConfigLoader('configs/srcnn.cfg', 'default')
    cfg = ConfigLoader('configs/srcnn.cfg', 'rvl_cdip')
    model = Model(cfg)
    model.train(srcnn_path=srcnn_path,
                random_scale=False, rotate=False, fliplr=False, fliptb=False)
    model.test(srcnn_path=srcnn_path)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # srcnn_path = "results/checkpoints/srresnet/srresnet_param_batch4_lr0.001_epoch3.pkl"
    
    # cfg = ConfigLoader('configs/srresnet.cfg', 'default')
    # cfg = ConfigLoader('configs/srresnet.cfg', 'rvl_cdip')
    # model = Model(cfg)
    # model.train(srcnn_path=srcnn_path,
    #             random_scale=False, rotate=False, fliplr=False, fliptb=False)
    # model.test(srcnn_path=srcnn_path)t