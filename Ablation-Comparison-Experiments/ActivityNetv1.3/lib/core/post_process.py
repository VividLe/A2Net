import torch
import pandas as pd
import numpy as np
import os
import json
import pickle

from core.nms import soft_nms, temporal_nms
from utils.utils import prepare_output_file

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


# TODO: what is the meaning of min_max_norm?
def min_max_norm(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def decode_prediction(df, cfg):
    '''
    decode predictions, fuse confidence score and classification score
    '''
    cls_scores_cls = [(df['score_' + str(i)]).values[:].tolist() for i in range(cfg.DATASET.NUM_CLASSES)]
    cls_scores_seg = [[cls_scores_cls[j][i] for j in range(cfg.DATASET.NUM_CLASSES)] for i in range(len(df))]

    # TODO: improve the efficiency of this, using pd operation
    cls_type_list = list()
    cls_score_list = list()
    for i in range(len(df)):
        # We exclude the background class, thus we should add 1 for the class index
        cls_score = np.array(cls_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        cls_score = cls_score.tolist()
        cls_score_list.append(max(cls_score))
        # determine the class idx
        cls_type = cls_score.index(max(cls_score)) + 1
        cls_type_list.append(cls_type)
    result_df = pd.DataFrame()
    result_df['cls_type'] = cls_type_list
    result_df['score'] = cls_score_list
    result_df['start'] = df.xmin.values[:]
    result_df['end'] = df.xmax.values[:]

    # TODO: why this processs?
    result_df['video_name'] = [df['video_name'].values[0] for _ in range(len(result_df))]
    return result_df


def post_process(df, epoch, cfg, is_soft_nms=False):
    # obtain video duration from annotation file
    annotation_file = os.path.join(cfg.ROOT_DIR, cfg.TEST.GT_FILE)
    gts = json.load(open(annotation_file, 'r'))
    gts = gts['database']

    result_file = os.path.join(cfg.ROOT_DIR, cfg.TEST.PREDICT_TXT_FILE+'_'+str(epoch).zfill(2)+'.json')
    if os.path.exists(result_file):
        os.remove(result_file)
    # load idx_name dictionary file
    idx_name_file = os.path.join(cfg.ROOT_DIR, cfg.TEST.IDX_NAME_FILE)
    idx_name = pickle.load(open(idx_name_file, 'rb'))

    # TODO: remove
    txt_file = os.path.join(cfg.ROOT_DIR,
                               cfg.TEST.PREDICT_TXT_FILE + '_' + str(epoch) + '.txt')
    if os.path.exists(txt_file):
        os.remove(txt_file)
    f = open(txt_file, 'a')

    result_dict = dict()

    # filter out confident background predictions
    # df = df[df.score_0 < cfg.TEST.FILTER_NEGATIVE_TH]
    # df = df[df.conf > cfg.TEST.FILTER_CONF_TH]
    video_name_list = list(set(df.video_name.values[:]))
    video_name_list.sort()

    for video_name in video_name_list:
        vid_duration = gts[video_name]['duration']
        tmp_df = df[df.video_name == video_name]
        if is_soft_nms:
            # df_pred = decode_prediction(tmp_df, cfg)
            df_nms = soft_nms(tmp_df, idx_name, cfg)
            df_vid = df_nms.sort_values(by='score', ascending=False)
        else:
            # df_pred = decode_prediction_naive(tmp_df, cfg)
            df_vid = temporal_nms(tmp_df, idx_name, cfg)

        detection_res = list()
        # for i in range(len(df_vid)):
        for i in range(min(cfg.TEST.TOP_K_RPOPOSAL, len(df_vid))):
            tmp_proposal = dict()
            tmp_proposal['score'] = df_vid.score.values[i]
            start_time = df_vid.start.values[i] * vid_duration
            end_time = df_vid.end.values[i] * vid_duration
            tmp_proposal['segment'] = [start_time, end_time]
            # tmp_proposal['label'] = df_vid.label.values[i]
            detection_res.append(tmp_proposal)
            # TODO: remove
            strout = '%s\t%.3f\t%.3f\t%.4f\n' % (
            video_name, float(start_time), float(end_time), df_vid.score.values[i])
            f.write(strout)
        result_dict[video_name] = detection_res
    # todo:remove
    f.close()

    # prepare for output file
    prepare_output_file(result_dict, result_file)


def decode_prediction_naive(df, cfg):
    '''
    select prediction with top confidence
    '''
    cls_scores_cls = [(df['score_' + str(i)]).values[:].tolist() for i in range(cfg.DATASET.NUM_CLASSES)]
    cls_scores_seg = [[cls_scores_cls[j][i] for j in range(cfg.DATASET.NUM_CLASSES)] for i in range(len(df))]

    # save top 3 prediction
    # TODO: improve this, we should save top predictions according to confidence of percentage
    # how many predictions do we use?
    cls_type_list = list()
    cls_score_list = list()
    for i in range(len(df)):
        cls_score = np.array(cls_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        cls_score = cls_score.tolist()
        cls_score_list.append(max(cls_score))
        # determine the class idx
        cls_type = cls_score.index(max(cls_score)) + 1
        cls_type_list.append(cls_type)
    result_df1 = pd.DataFrame()
    result_df1['out_type'] = cls_type_list
    result_df1['out_score'] = cls_score_list
    result_df1['start'] = df.xmin.values[:]
    result_df1['end'] = df.xmax.values[:]

    # append the second largest
    cls_type_list = list()
    cls_score_list = list()
    for i in range(len(df)):
        cls_score = np.array(cls_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        cls_score = cls_score.tolist()
        # different here
        cls_score[cls_score.index(max(cls_score))] = 0
        cls_score_list.append(max(cls_score))
        cls_type = cls_score.index(max(cls_score)) + 1
        cls_type_list.append(cls_type)
    result_df2 = pd.DataFrame()
    result_df2['out_type'] = cls_type_list
    result_df2['out_score'] = cls_score_list
    result_df2['start'] = df.xmin.values[:]
    result_df2['end'] = df.xmax.values[:]
    result_df1 = pd.concat([result_df1, result_df2])

    # append the third largest
    cls_type_list = list()
    cls_score_list = list()
    for i in range(len(df)):
        cls_score = np.array(cls_scores_seg[i][1:]) * min_max_norm(df.conf.values[i])
        cls_score = cls_score.tolist()
        # different here
        cls_score[cls_score.index(max(cls_score))] = 0
        cls_score[cls_score.index(max(cls_score))] = 0
        cls_score_list.append(max(cls_score))
        cls_type = cls_score.index(max(cls_score)) + 1
        cls_type_list.append(cls_type)
    result_df2 = pd.DataFrame()
    result_df2['out_type'] = cls_type_list
    result_df2['out_score'] = cls_score_list
    result_df2['start'] = df.xmin.values[:]
    result_df2['end'] = df.xmax.values[:]
    result_df1 = pd.concat([result_df1, result_df2])

    result_df1['video_name'] = [df['video_name'].values[0] for _ in range(len(result_df1))]
    return result_df1


