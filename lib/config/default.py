from easydict import EasyDict as edict
import yaml
import numpy as np

config = edict()

config.BASIC = edict()
config.BASIC.CFG_FILE = ''
config.BASIC.LOG_DIR = ''
config.BASIC.ROOT_DIR = ''
config.BASIC.WORKERS = 1
config.BASIC.PIN_MEMORY = True
config.BASIC.SHOW_CFG = False
config.BASIC.SEED = 0
config.BASIC.SAVE_PREDICT_RESULT = True
config.BASIC.BACKUP_CODES = True
config.BASIC.BACKUP_LISTS = ['lib']
config.BASIC.CREATE_OUTPUT_DIR = True

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLE = True

# DATASET related params
config.DATASET = edict()
config.DATASET.TRAIN_SPLIT = ''
config.DATASET.VAL_SPLIT = ''
config.DATASET.NUM_CLASSES = 21
config.DATASET.CLASS_REAL = (1,)
config.DATASET.OVERLAP_RATIO_TH = 0.9
config.DATASET.MAX_SEGMENT_NUM = 1
config.DATASET.FEAT_DIR = ''
config.DATASET.RESCALE_TEM_LENGTH = 512
config.DATASET.CLASS_IDX = [0]
config.DATASET.WINDOW_SIZE = 1
config.DATASET.PIN_MEMORY = False

# MODEL related params
config.MODEL = edict()
config.MODEL.IN_FEAT_DIM = 512
config.MODEL.FEATURE_DIM = 512
config.MODEL.MID_FEAT_DIM = 512
config.MODEL.ANCHOR_FEAT_DIM = 512
config.MODEL.LAYERS_NAME = ['AL1']
config.MODEL.LAYERS_NAME_AB = ['AL1']
config.MODEL.LAYERS_NAME_AF = ['AL1']
config.MODEL.SCALE = {'AL1': 0.0625}
config.MODEL.NUM_ANCHORS = {'AL1': 16,}
config.MODEL.ASPECT_RATIOS = {'AL1': [0.5]}
config.MODEL.NUM_DBOX = {'AL1': 5}
config.MODEL.LOCAL_MASK_WIDTH = 3
config.MODEL.BASE_FEAT_DIM = 1
config.MODEL.CON1_FEAT_DIM = 1
config.MODEL.CON2_FEAT_DIM = 1
config.MODEL.CON3_FEAT_DIM = 1
config.MODEL.CON4_FEAT_DIM = 1
config.MODEL.CON5_FEAT_DIM = 1
config.MODEL.CON6_FEAT_DIM = 1
config.MODEL.CON7_FEAT_DIM = 1
config.MODEL.REDU_CHA_DIM = 1
config.MODEL.RESCALE = True
config.MODEL.TEMPORAL_LENGTH = 1
config.MODEL.INPUT_TEM_LENGTH = 1
config.MODEL.TEMPORAL_STRIDE = 1

# TRAIN related params
config.TRAIN = edict()
config.TRAIN.LR = 0.0001
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.BEGIN_EPOCH = 1
config.TRAIN.END_EPOCH = 1
config.TRAIN.P_CLASS_AB = 1
config.TRAIN.P_LOC_AB = 1
config.TRAIN.P_CONF_AB = 1
config.TRAIN.P_CLASS_AF = 1
config.TRAIN.P_LOC_AF = 1
config.TRAIN.NEGATIVE_RATIO = 1
config.TRAIN.ANCHOR_RX_SCALE = 0.1
config.TRAIN.ANCHOR_RW_SCALE = 0.1
config.TRAIN.MATCH_TH = 0.5
config.TRAIN.MODELS_DIR = ''
config.TRAIN.MODE = ''
config.TRAIN.LOG_FILE = ''
config.TRAIN.MODEL_DIR = ''
config.TRAIN.LR_DECAY_EPOCHS = [1]
config.TRAIN.LR_DECAY_FACTOR = 0.1

# TEST related params
config.TEST = edict()
config.TEST.BATCH_SIZE = 1
config.TEST.NMS_TH = 0.1
config.TEST.FILTER_NEGATIVE_TH = 0.98
config.TEST.FILTER_CONF_TH = 0.1
config.TEST.OUTDF_COLUMNS_AB = ['', ]
config.TEST.OUTDF_COLUMNS_AF = ['', ]
config.TEST.PREDICT_CSV_FILE = ''
config.TEST.PREDICT_TXT_FILE = ''
config.TEST.VIDEOS_25FPS = ['']
config.TEST.VIDEOS_24FPS = ['']
config.TEST.MODEL_PATH = ''
config.TEST.FRAME_RATE = 25
config.TEST.IDX_NAME_FILE = ''
config.TEST.TOP_K_RPOPOSAL = 1
config.TEST.SOFT_NMS_ALPHA = 0.7
config.TEST.SOFT_NMS_LOW_TH = 0.6
config.TEST.SOFT_NMS_HIGH_TH = 0.9
config.TEST.GT_FILE = ''
config.TEST.EVAL_INTERVAL = 1
config.TEST.CATE_IDX_OCC = 1
config.TEST.CATE_IDX_REP = 1
config.TEST.CONCAT_AB = 1
config.TEST.OUTDF_COLUMNS = ['video_name']


def genconfigonfig(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


if __name__ == '__main__':
    cfg_file = '/data/home/v-yale/ActionLocalization/experiments/thumos/dssad.yaml'
    update_config(cfg_file)
    print(config)
