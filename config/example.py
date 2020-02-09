from config import get_cfg_defaults  # local variable usage pattern, or:
# from config import cfg  # global singleton usage pattern


if __name__ == "__main__":
  """  Put the following code in the file that needs to call the parameters  """
  cfg = get_cfg_defaults()
  cfg.merge_from_file("example.yaml")
  cfg.freeze()
  print(cfg.THRESHOLD.MAXSIZE)  # Example of how to call