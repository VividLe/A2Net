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


def min_max_norm(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def result_process_ab(video_names, video_len, start_frames, anchors_class, anchors_overlap, anchors_xmin, anchors_xmax, cfg):
    out_df = pd.DataFrame()

    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, video_len)
    xmaxs = xmaxs + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    # select score0, for filter
    anchors_score0 = anchors_class[:, :, 0]
    anchors_score0_tmp = np.reshape(anchors_score0, num_element)
    out_df['score_0'] = anchors_score0_tmp

    # record confidence score and category index
    scores_action = anchors_class[:, :, 1:]
    # calculate confidence score: jointly consider overlap and classification score
    overlap_norm = min_max_norm(anchors_overlap)
    scores_action = scores_action * np.expand_dims(overlap_norm, axis=2)

    max_values = np.amax(scores_action, axis=2)
    conf_tmp = np.reshape(max_values, num_element)
    out_df['conf'] = conf_tmp
    max_idxs = np.argmax(scores_action, axis=2)
    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    for i in range(len(cate_idx_tmp)):
        cate_idx_tmp[i] = class_real[int(cate_idx_tmp[i])]
    out_df['cate_idx'] = cate_idx_tmp

    # filter
    out_df = out_df[out_df.score_0 < cfg.TEST.FILTER_NEGATIVE_TH]
    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AB]

    return out_df


# Notice: maybe merge this with result_process_ab
def result_process_af(video_names, start_frames, cls_scores, anchors_xmin, anchors_xmax, cfg):
    out_df = pd.DataFrame()

    layer0_name = cfg.MODEL.LAYERS_NAME[0]
    feat_tem_width = cfg.MODEL.TEMPORAL_LENGTH[layer0_name]
    frame_window_width = cfg.DATASET.WINDOW_SIZE

    xmins = np.maximum(anchors_xmin, 0)
    xmins = xmins / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)
    xmaxs = np.minimum(anchors_xmax, feat_tem_width)
    xmaxs = xmaxs / feat_tem_width * frame_window_width + np.expand_dims(start_frames, axis=1)

    # expand video_name
    vid_name_df = list()
    num_tem_loc = anchors_xmin.shape[1]
    for i in range(len(video_names)):
        vid_names = [video_names[i]] * num_tem_loc
        vid_name_df.extend(vid_names)
    out_df['video_name'] = vid_name_df

    # reshape numpy array
    # Notice: this is not flexible enough
    num_element = anchors_xmin.shape[0] * anchors_xmin.shape[1]
    xmins_tmp = np.reshape(xmins, num_element)
    out_df['xmin'] = xmins_tmp
    xmaxs_tmp = np.reshape(xmaxs, num_element)
    out_df['xmax'] = xmaxs_tmp

    scores_action = cls_scores[:, :, 1:]
    max_values = np.amax(scores_action, axis=2)
    conf_tmp = np.reshape(max_values, num_element)
    out_df['conf'] = conf_tmp
    max_idxs = np.argmax(scores_action, axis=2)
    max_idxs = max_idxs + 1
    cate_idx_tmp = np.reshape(max_idxs, num_element)
    # Notice: convert index into category type
    class_real = cfg.DATASET.CLASS_IDX
    for i in range(len(cate_idx_tmp)):
        cate_idx_tmp[i] = class_real[int(cate_idx_tmp[i])]
    out_df['cate_idx'] = cate_idx_tmp

    out_df = out_df[cfg.TEST.OUTDF_COLUMNS_AB]
    return out_df
