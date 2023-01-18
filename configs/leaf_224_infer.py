from yacs.config import CfgNode as CN


"""
_C.MODE (str): MODE should be one of 'train', 'eval', 'infer'
_C.INFER_LOGIT (bool): do not pass output through softmax if True

_C.NUM_CLASSES (int): number of classes
_C.NUM_FEATURES (int): extractor's output size
_C.RESIZE (tuple): size to be resize, [H, W]
_C.NORMALIZE (tuple): [mean, std]

_C.EXT_LR (float): learning rate of extractor
_C.CLF_LR (float): learning rate of classifier
_C.EXT_WEIGHT_DECAY (float): weight decay of extractor
_C.CLF_WEIGHT_DECAY (float): weight decay of classifier
_C.ETA_MIN (float): minimum learning rate
_C.CLIP_FACTOR (float): gradient clipping factor
_C.VAL_BATCH_SIZE (int): batch size when validation
_C.NUM_WORKERS (int): number of dataloader workers
_C.PATCH_SIZE (int): size of patch, int
_C.MEAN_STD (Optional[tuple]): patch filter threshold, [mean, std]
_C.PATCH_BATCH_SIZE (int): size of patch batch

_C.SHUFFLE (bool): if False, patch will be grouped into patch batch in order
_C.RESUME_PATH (Optional[str]): checkpoint file path
_C.BEST_SCORE (int): best score to be compared
_C.SCORE_WINDOW (int): metrices are averaged within window, unit of window is iteration
to be exact, FREQ means period not frequency
_C.SAVE_FREQ (int): how often save checkpoint. the unit is epoch
_C.SAVE_ITER_FREQ (int): independent with SAVE_FREQ. the unit is iteration
_C.SCORE_UPDATE_FREQ (int): how often record score to history. the unit is iteration

_C.TRAIN_ONLY_CLASSIFIER (bool): do not train extractor, if True
_C.LOAD_OPTIMIZER (bool): load optimizer when resuming with checkpoint, if True
_C.LOAD_SCHEDULER (bool): load scheduler when resuming with checkpoint, if True
_C.LOAD_HISTORY (bool): load history when resuming with checkpoint, if True
_C.EXPERIMENT_NAME (str): name for distinguishing folders that contain checkpoints

_C.DATA.IS_MULTI_LABEL (bool): is data multi labeled
_C.DATA.READ_TYPE (str): format in which the data is organized, 'csv' or ''
# Following two are not needed if READ_TYPE is not 'csv'
_C.DATA.IMG_PATH_COL (Optional[str]): column name for image path in csv file
_C.DATA.LABEL_COL (Optional[str]): column name for label in csv file

_C.DATA.PATH.TRAIN_DIR (str): directory path in which train data stored
_C.DATA.PATH.VAL_DIR (Optional[str]): directory path in which validation data stored
_C.DATA.PATH.TEST_DIR (Optional[str]): directory path in which test data stored
# if DATA.READ_TYPE is 'csv', DATA.PATH.TRAIN_CSV must be provided because this is used to get class_to_idx
_C.DATA.PATH.TRAIN_CSV (Optional[str]): train csv path
_C.DATA.PATH.VAL_CSV (Optional[str]): validation csv path
_C.DATA.PATH.TEST_CSV (Optional[str]): test csv path
_C.DATA.PATH.RECORDER (str): recording folder name

# Korea Standard Time (int): 9
# Eastern Standard Time (int): -5
_C.TIME_ZONE (int): time zone, it is used in naming folder name
"""

_C = CN()

_C.MODE = 'infer'
_C.INFER_LOGIT = True

_C.NUM_CLASSES = 6
_C.NUM_FEATURES = 3072
_C.RESIZE = [2688, 4032]
_C.NORMALIZE = [[0.48738927, 0.62685798, 0.40773353], 
                [0.18765608, 0.16413330, 0.19589159]]

_C.EXT_LR = 1e-4
_C.CLF_LR = 1e-4
_C.EXT_WEIGHT_DECAY = 1e-2
_C.CLF_WEIGHT_DECAY = 1e-2
_C.ETA_MIN = 1e-5
_C.CLIP_FACTOR = 0.04
_C.EPOCHS = 1
_C.BATCH_SIZE = 8
_C.VAL_BATCH_SIZE = 10
_C.NUM_WORKERS = 1
_C.PATCH_SIZE = 224
_C.MEAN_STD = None
_C.PATCH_BATCH_SIZE = 27

_C.SHUFFLE = False
_C.RESUME_PATH = 'results/dense_224/2022_12_22_07_38_44/last_iter__train_save.pth'
_C.BEST_SCORE = 0
_C.SCORE_WINDOW = 25
_C.SAVE_FREQ = 1
_C.SAVE_ITER_FREQ = 10
_C.SCORE_UPDATE_FREQ = 1

_C.TRAIN_ONLY_CLASSIFIER = False
_C.LOAD_OPTIMIZER = True
_C.LOAD_SCHEDULER = False
_C.LOAD_HISTORY = False
_C.EXPERIMENT_NAME = 'dense_224'

_C.DATA = CN()
_C.DATA.IS_MULTI_LABEL = True
_C.DATA.READ_TYPE = 'csv' 
_C.DATA.IMG_PATH_COL = 'image'
_C.DATA.LABEL_COL = 'labels'

_C.DATA.PATH = CN()
_C.DATA.PATH.TRAIN_DIR = None
_C.DATA.PATH.VAL_DIR = None
_C.DATA.PATH.TEST_DIR = 'data/train_images'
_C.DATA.PATH.TRAIN_CSV = 'data/train_labels.csv'
_C.DATA.PATH.VAL_CSV = None
_C.DATA.PATH.TEST_CSV = 'data/val_labels.csv'
_C.DATA.PATH.RECORDER = 'results'

_C.TIME_ZONE = 9

cfg = _C