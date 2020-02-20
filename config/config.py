from yacs.config import CfgNode as CN

_C = CN()

_C.BASE = CN()
# Number of GPUS to use in the experiment
_C.BASE.MODEL = 'ModelName'
_C.BASE.VERSION = 'v1'
_C.BASE.ENV = 'EnvironmentName'
_C.BASE.TASK_ID = '12345'
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
_C.ADDRESS.DETECTION.GT_JSON_DIR = ''

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
_C.ADDRESS.RESULT_DIR = ''
_C.ADDRESS.CACHE_DIR = ''
_C.ADDRESS.LOGGER_DIR = ''
_C.ADDRESS.SUMMARY_DIR = ''

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
_C.MODEL.DECAY_STEP = 45
_C.MODEL.OPTIMIZER = 'Adam'
_C.MODEL.LOSS = ''
_C.MODEL.PATIENCE = 5
_C.MODEL.INIT_TYPE = 'xavier'
_C.MODEL.lambda_inside_score_loss = 4.0
_C.MODEL.lambda_side_vertex_code_loss = 1.0
_C.MODEL.lambda_side_vertex_coord_loss = 1.0
_C.MODEL.shrink_ratio = 0.2
_C.MODEL.shrink_side_ratio = 0.6
_C.MODEL.epsilon = 1e-4
_C.MODEL.pixel_size = 4

_C.MODEL.DETAILS = CN()
_C.MODEL.DETAILS.HIDDEN_LAYER = 10

_C.THRESHOLD = CN()
_C.THRESHOLD.MAXSIZE = 100
_C.THRESHOLD.MINSIZE = 1
_C.THRESHOLD.TEXT_SCALE = 75
_C.THRESHOLD.iou_threshold = 0.5
_C.THRESHOLD.pixel_threshold = 0.9
_C.THRESHOLD.side_vertex_pixel_threshold = 0.9
_C.THRESHOLD.trunc_threshold = 0.1

_C.SAVE_FREQ = 4000
_C.SHOW_FREQ = 100
_C.VAL_FREQ = 1000

_C.predict_cut_text_line = False
_C.draw = 'store_true'

_C.targetH = 32
_C.targetW = 100
_C.BidirDecoder = True
_C.inputDataType = 'torch.cuda.FloatTensor'
_C.maxBatch = 256
_C.CUDA = True

_C.TEXTSNAKE = CN()
_C.TEXTSNAKE.input_size = 512
_C.TEXTSNAKE.exp_name = 'example1'
_C.TEXTSNAKE.output_channel = 1

  # train opts
_C.TEXTSNAKE.start_iter = 0
_C.TEXTSNAKE.max_iters = 50000
_C.TEXTSNAKE.lr_adjust = 'fix'
_C.TEXTSNAKE.stepvalues = []
_C.TEXTSNAKE.weight_decay = '0.'
_C.TEXTSNAKE.wd = '0.'
_C.TEXTSNAKE.gamma = 0.1
_C.TEXTSNAKE.momentum = 0.9

  # data args
_C.TEXTSNAKE.rescale = 255.0
_C.TEXTSNAKE.means = [0.474, 0.445, 0.418]
_C.TEXTSNAKE.stds = [1., 1., 1.]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`


