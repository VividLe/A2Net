import argparse
import os
import pickle
import numpy as np
import pandas as pd
import json


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subset', default='validation')
    parser.add_argument('-window_size', default=64)
    parser.add_argument('-overlap_ratio_threshold', default=0.9)
    # todo: modify
    parser.add_argument('-is_train', default=False)
    parser.add_argument('-gt_file', default='/disk/yangle/AbHacs/lib/dataset/materials/HACS_segments_v1.1.1.json')
    parser.add_argument('-spa_feature_dir', default='/disk/yangle/AbHacs/dataset/feature/slowfast')
    parser.add_argument('-res_file', default='/disk/yangle/AbHacs/dataset/feature/anns_val.pkl')
    args = parser.parse_args()
    return args


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feat_sf']
    return feature


def window_data(start_idx, action_boundaries, args):
    window_size = args.window_size
    end_idx = start_idx + window_size

    box = list()
    window = [start_idx, end_idx]

    for act in action_boundaries:
        act_start = act[0]
        act_end = act[1]
        if act_end <= act_start:
            continue
        # assert act_end > act_start

        interaction = min(end_idx, act_end) - max(start_idx, act_start)
        overlap_ration = interaction * 1.0 / (act_end - act_start)

        if overlap_ration > args.overlap_ratio_threshold:
            gt_start = max(start_idx, act_start) - start_idx
            gt_end = min(end_idx, act_end) - start_idx
            box.append([gt_start*1.0/window_size, gt_end*1.0/window_size, overlap_ration])

    box = np.array(box).astype('float32')
    return box, window


def sliding_window(vid_name, vid_anns, args):
    window_size = args.window_size
    if args.is_train:
        stride = window_size / 4
    else:
        stride = window_size / 2

    # read feature
    feat_file = os.path.join(args.spa_feature_dir, 'v_'+vid_name+'.npz')
    if not os.path.exists(feat_file):
        return [], [], []

    feature = load_npz_feat(feat_file)
    # the shortest temporal length should be 256
    tem_length = max(feature.shape[0], args.window_size)

    vid_dur = float(vid_anns['duration'])
    anns = vid_anns['annotations']

    action_boundaries = list()
    # todo: update here
    cate_label = anns[0]['label']
    for ann in anns:
        segment = ann['segment']
        start_idx = segment[0] / vid_dur * tem_length
        end_idx = segment[1] / vid_dur * tem_length
        action_boundaries.append([start_idx, end_idx])

    num_window = int(1.0*(tem_length + stride - window_size) / stride)
    windows_start = [i * stride for i in range(num_window)]

    if tem_length > window_size:
        windows_start.append(tem_length - window_size)

    label = list()
    box = list()
    window = list()
    for start in windows_start:
        if args.is_train:
            box_tmp, window_tmp = window_data(start, action_boundaries, args)
            if box_tmp.shape[0] > 0:
                box.append(box_tmp)
                window.append(window_tmp)
                label.append(cate_label)
        else:
            window.append([int(start), int(start + window_size)])
    return window, label, box


def gene_info(args):
    with open(args.gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    anno = dict()
    num_instance = 0
    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] != args.subset:
            continue

        gt_window, gt_label, gt_box = sliding_window(vid_name, vid_anns, args)
        if len(gt_window) > 0:
            anno[vid_name] = [gt_window, gt_label, gt_box]
            num_instance += len(gt_window)

    print('total samples:', len(list(anno.keys())))
    print('total windows:', num_instance)

    with open(args.res_file, 'wb') as f:
        pickle.dump(anno, f)


if __name__ == '__main__':

    args = args_parser()
    gene_info(args)



