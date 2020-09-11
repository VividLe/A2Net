import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.init as init

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()


def jaccard_with_anchors(anchors_min, anchors_max, box_min, box_max):
    '''
    calculate tIoU for anchors and ground truth action segment
    '''
    inter_xmin = torch.max(anchors_min, box_min)
    inter_xmax = torch.min(anchors_max, box_max)
    inter_len = inter_xmax - inter_xmin

    inter_len = torch.max(inter_len, torch.tensor(0.0).type_as(dtype))
    union_len = anchors_max - anchors_min - inter_len + box_max - box_min

    jaccard = inter_len / union_len
    return jaccard


def tiou(anchors_min, anchors_max, len_anchors, box_min, box_max):
    '''
    calculate jaccatd score between a box and an anchor
    '''
    inter_xmin = np.maximum(anchors_min, box_min)
    inter_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(inter_xmax-inter_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    tiou = np.divide(inter_len, union_len)
    return tiou


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()


def result_process_ab(video_names, start_frames, anchors_overlap, xmin, xmax, cfg):
    out_df = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS)
    for i in range(len(video_names)):
        tmp_df = pd.DataFrame()
        video_name = video_names[i]
        start_frame = int(start_frames[i])
        overlap = anchors_overlap[i, :]

        num_anchor = len(xmax)

        tmp_df['video_name'] = [video_name] * num_anchor
        tmp_df['start'] = [start_frame] * num_anchor
        tmp_df['conf'] = overlap
        tmp_df['xmin'] = xmin
        tmp_df['xmax'] = xmax

        tmp_df.xmin = np.maximum(tmp_df.xmin, 0)
        tmp_df.xmax = np.minimum(tmp_df.xmax, 1)
        tmp_df.xmin = tmp_df.xmin + tmp_df.start
        tmp_df.xmax = tmp_df.xmax + tmp_df.start

        out_df = pd.concat([out_df, tmp_df])
    return out_df

def result_process_af(video_names, start_frames, conf_scores, xmins, xmaxs, cfg):
    out_df = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS)
    for i in range(len(video_names)):
        tmp_df = pd.DataFrame()
        video_name = video_names[i]
        start_frame = int(start_frames[i])
        score = conf_scores[i, :]  # [254]
        xmin = xmins[i, :]
        xmax = xmaxs[i, :]

        # constrain action prediction in range [0, 128]
        xmin = np.maximum(xmin, 0)
        xmax = np.minimum(xmax, cfg.MODEL.INPUT_TEM_LENGTH)
        # convert relative location to [0, 1]
        xmin = xmin / cfg.MODEL.INPUT_TEM_LENGTH
        xmax = xmax / cfg.MODEL.INPUT_TEM_LENGTH

        num_preds = len(xmax)

        tmp_df['video_name'] = [video_name] * num_preds
        tmp_df['start'] = [start_frame] * num_preds
        tmp_df['conf'] = score
        tmp_df['xmin'] = xmin
        tmp_df['xmax'] = xmax

        # filter out confident background predictions#########################################################
        # tmp_df = tmp_df[tmp_df.score_0 < cfg.TEST.FILTER_NEGATIVE_TH]

        out_df = pd.concat([out_df, tmp_df])
    return out_df
