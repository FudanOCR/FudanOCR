from config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
  """  将config.py文件import进需要调用参数的文件后加入下述代码  """
  cfg = get_cfg_defaults()
  cfg.merge_from_file("config.yaml")  # 将不同的配置信息写入yaml，覆盖默认配置
  cfg.freeze()  # 合并
  print(cfg.THRESHOLD.MAXSIZE)  # 调用方式