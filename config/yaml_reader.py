from yacs.config import CfgNode as CN

def read_config_file(config_file):
    # 用yaml重构配置文件
    f = open(config_file)
    opt = CN.load_cfg(f)
    return opt