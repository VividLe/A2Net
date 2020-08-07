import numpy as np
import pandas as pd

from core.utils_ab import tiou


def temporal_nms(df, cfg):
    '''
    temporal nms
    I should understand this process
    '''

    type_set = list(set(df.cate_idx.values[:]))
    # type_set.sort()

    # returned values
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

    # attention: for THUMOS, a sliding window may contain actions from different class
    for t in type_set:
        label = t
        tmp_df = df[df.cate_idx == t]

        start_time = np.array(tmp_df.xmin.values[:])
        end_time = np.array(tmp_df.xmax.values[:])
        scores = np.array(tmp_df.conf.values[:])

        duration = end_time - start_time
        order = scores.argsort()[::-1]

        keep = list()
        while (order.size > 0) and (len(keep) < cfg.TEST.TOP_K_RPOPOSAL):
            i = order[0]
            keep.append(i)
            tt1 = np.maximum(start_time[i], start_time[order[1:]])
            tt2 = np.minimum(end_time[i], end_time[order[1:]])
            intersection = tt2 - tt1
            union = (duration[i] + duration[order[1:]] - intersection).astype(float)
            iou = intersection / union

            inds = np.where(iou <= cfg.TEST.NMS_TH)[0]
            order = order[inds + 1]

        # record the result
        for idx in keep:
            rlabel.append(label)
            rstart.append(float(start_time[idx]))
            rend.append(float(end_time[idx]))
            rscore.append(scores[idx])

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df


def soft_nms(df, idx_name, cfg):
    df = df.sort_values(by='score', ascending=False)
    save_file = '/data/home/v-yale/ActionLocalization/output/df_sort.csv'
    df.to_csv(save_file, index=False)

    tstart = list(df.start.values[:])
    tend = list(df.end.values[:])
    tscore = list(df.score.values[:])
    tcls_type = list(df.cls_type.values[:])
    rstart = list()
    rend = list()
    rscore = list()
    rlabel = list()

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
        # video class label
        cls_type = tcls_type[max_idx]
        label = idx_name[cls_type]
        rlabel.append(label)

        tstart.pop(max_idx)
        tend.pop(max_idx)
        tscore.pop(max_idx)
        tcls_type.pop(max_idx)

    new_df = pd.DataFrame()
    new_df['start'] = rstart
    new_df['end'] = rend
    new_df['score'] = rscore
    new_df['label'] = rlabel
    return new_df

