from yacs.config import CfgNode as CN

_C = CN()

_C.BASE = CN()
# Number of GPUS to use in the experiment
_C.BASE.MODEL = 'ModelName'
_C.BASE.VERSION = 'v1'
_C.BASE.ENV = 'EnvironmentName'
_C.BASE.TASK_ID = 12345
_C.BASE.NETWORK = 'NetworkName'
_C.BASE.NUM_GPUS = 1
_C.BASE.GPU_ID = [0]
# Number of workers for doing things
_C.BASE.WORKERS = 4

_C.DATASETS = CN()
_C.DATASETS.DATASET = 'ICDAR15'
_C.DATASETS.TYPE = 'DataType'

_C.ADDRESS = CN()

_C.ADDRESS.DETECTION = CN()
_C.ADDRESS.DETECTION.TRAIN_DATA_DIR = ''
_C.ADDRESS.DETECTION.TRAIN_GT_DIR = ''
_C.ADDRESS.DETECTION.TEST_DATA_DIR = ''
_C.ADDRESS.DETECTION.TEST_GT_DIR = ''
_C.ADDRESS.DETECTION.VAL_DATA_DIR = ''
_C.ADDRESS.DETECTION.VAL_GT_DIR = ''

_C.ADDRESS.RECOGNITION = CN()
_C.ADDRESS.RECOGNITION.ALPHABET = ''
_C.ADDRESS.RECOGNITION.TRAIN_DATA_DIR = ''
_C.ADDRESS.RECOGNITION.TRAIN_LABEL_DIR = ''
_C.ADDRESS.RECOGNITION.TEST_DATA_DIR = ''
_C.ADDRESS.RECOGNITION.TEST_LABEL_DIR = ''
_C.ADDRESS.RECOGNITION.VAL_DATA_DIR = ''
_C.ADDRESS.RECOGNITION.VAL_LABEL_DIR = ''

_C.ADDRESS.CHECKPOINTS_DIR = ''
_C.ADDRESS.PRETRAIN_MODEL_DIR = ''
_C.ADDRESS.CACHE_DIR = ''
_C.ADDRESS.LOGGER_DIR = ''

_C.IMAGE = CN()
_C.IMAGE.IMG_H = 32
_C.IMAGE.IMG_W = 100
_C.IMAGE.IMG_CHANNEL = 3

_C.FUNCTION = CN()
_C.FUNCTION.VAL_ONLY = False
_C.FUNCTION.FINETUNE = False

_C.MODEL = CN()
_C.MODEL.EPOCH = 100
_C.MODEL.BATCH_SIZE = 4
_C.MODEL.LR = 0.01
_C.MODEL.DECAY_RATE = 0.1
_C.MODEL.OPTIMIZER = 'Adam'
_C.MODEL.LOSS = ''

_C.MODEL.DETAILS = CN()
_C.MODEL.DETAILS.HIDDEN_LAYER = 10

_C.THRESHOLD = CN()
_C.THRESHOLD.MAXSIZE = 100
_C.THRESHOLD.MINSIZE = 1
_C.THRESHOLD.TEXT_SCALE = 75

_C.SAVE_FREQ = 4000
_C.SHOW_FREQ = 100
_C.VAL_FREQ = 1000

_C.targetH= 32
_C.targetW=100
_C.BidirDecoder= True
_C.inputDataType= 'torch.cuda.FloatTensor'
_C.maxBatch= 256
_C.CUDA= True


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`


