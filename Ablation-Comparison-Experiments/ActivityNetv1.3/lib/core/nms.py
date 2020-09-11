import numpy as np
import pandas as pd

from core.anchor_box_utils import tiou


# def temporal_nms(df, idx_name, cfg):
#     '''
#     temporal nms
#     I should understand this process
#     '''
#
#     type_set = list(set(df.out_type.values[:]))
#     type_set.sort()
#
#     # returned values
#     rstart = list()
#     rend = list()
#     rscore = list()
#
#     for t in type_set:
#         tmp_df = df[df.out_type == t]
#
#         start_time = np.array(tmp_df.start.values[:])
#         end_time = np.array(tmp_df.end.values[:])
#         scores = np.array(tmp_df.out_score.values[:])
#         tmp_type = list(tmp_df.out_type.values[:])
#         duration = end_time - start_time
#         order = scores.argsort()[::-1]
#
#         keep = list()
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
#             # TODO: check shape here
#             tt1 = np.maximum(start_time[i], start_time[order[1:]])
#             tt2 = np.minimum(end_time[i], end_time[order[1:]])
#             intersection = tt2 - tt1
#             union = (duration[i] + duration[order[1:]] - intersection).astype(float)
#             iou = intersection / union
#
#             inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
#             order = order[inds + 1]
#
#         for idx in keep:
#             # skip 0, amgibuous/background class
#             if tmp_type[idx] in list(range(1, cfg.DATASET.NUM_CLASSES)):
#                 rstart.append(float(start_time[idx]))
#                 rend.append(float(end_time[idx]))
#                 rscore.append(scores[idx])
#
#     new_df = pd.DataFrame()
#     new_df['start'] = rstart
#     new_df['end'] = rend
#     new_df['score'] = rscore
#     return new_df


def temporal_nms(df, idx_name, cfg):
    '''
    temporal nms
    '''

    # returned values
    rstart = list()
    rend = list()
    rscore = list()

    start_time = np.array(df.xmin.values[:])
    end_time = np.array(df.xmax.values[:])
    scores = np.array(df.conf.values[:])
    duration = end_time - start_time
    order = scores.argsort()[::-1]

    keep = list()
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(start_time[i], start_time[order[1:]])
        tt2 = np.minimum(end_time[i], end_time[order[1:]])
        intersection = tt2 - tt1
        union = (duration[i] + duration[order[1:]] - intersection).astype(float)
        iou = intersection / union

        inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
        order = order[inds + 1]

    for idx in keep:
        rstart.append(float(start_time[idx]))
        rend.append(float(end_time[idx]))
        rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    return new_df


def soft_nms(df, idx_name, cfg):
    df = df.sort_values(by='conf', ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.conf.values[:])
    rstart = list()
    rend = list()
    rscore = list()

    while len(tscore) > 0 and len(rscore) <= cfg.TEST.TOP_K_RPOPOSAL:
        max_idx = np.argmax(tscore)
        tmp_width = tend[max_idx] - tstart[max_idx]
        iou = tiou(tstart[max_idx], tend[max_idx], tmp_width, np.array(tstart), np.array(tend))
        iou_exp = np.exp(-np.square(iou) / cfg.TEST.SOFT_NMS_ALPHA)
        for idx in range(len(tscore)):
            if idx != max_idx:
                tmp_iou = iou[idx]
                threshold = cfg.TEST.SOFT_NMS_LOW_TH + (cfg.TEST.SOFT_NMS_HIGH_TH - cfg.TEST.SOFT_NMS_LOW_TH) * tmp_width
                if tmp_iou > threshold:
                    tscore[idx] = tscore[idx] * iou_exp[idx]
        rstart.append(tstart[max_idx])
        rend.append(tend[max_idx])
        rscore.append(tscore[max_idx])

        tstart.pop(max_idx)
        tend.pop(max_idx)
        tscore.pop(max_idx)

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    return new_df

