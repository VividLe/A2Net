import argparse
import json
import os
import numpy as np
import pickle
import cv2
import pandas as pd
import sys
sys.path.append('/data/yangle/ActionLocalization/lib')
from config import cfg, update_config
from core.prediction_box_match import location_point_back


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subset', default='training')
    parser.add_argument('-gt_file', default='../materials/activity_net.v1-3.min.json')
    parser.add_argument('-tem_length', default=128)
    parser.add_argument('-feature_size', default=(128, 2048))
    parser.add_argument('-cfg_file', default='/data/yangle/ActionLocalization/experiments/anet/af.yaml')
    parser.add_argument('-feat_spa_dir', default='/data/yangle/ExtractSlowFastFeature/result/feature')
    parser.add_argument('-feat_tem_dir', default='')
    parser.add_argument('-res_dir', default='/data/yangle/ActionLocalization/data/binary_cls_proposal')
    return parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    frame_cnt = datas['frame_cnt']
    return feature, frame_cnt


def idx_in_actions(idx, act_range, bdys):
    '''
    assign action instance to proper layers according to act_length
    bdys should in range [0, 128]
    '''
    for bound in bdys:
        gt_length = bound[1] - bound[0]
        if (gt_length < act_range[0]) or (gt_length > act_range[1]):
            continue
        if (idx >= bound[0]) and (idx <= bound[1]):
            reg_left = idx - bound[0]
            reg_right = bound[1] - idx
            return True, reg_left, reg_right
    return False, 0, 0


# currently, each action instance only match one positive samples
def boundary_to_label(bdys, locs, act_ranges):
    cls_labels = np.zeros(len(locs), dtype=np.int)
    reg_labels = np.zeros((len(locs), 2), dtype=np.float)

    i = 0
    for idx, act_range in zip(locs, act_ranges):
        flag, reg_left, reg_right = idx_in_actions(idx, act_range, bdys)
        if flag:
            cls_labels[i] = 1
            reg_labels[i, 0] = reg_left
            reg_labels[i, 1] = reg_right
        else:
            cls_labels[i] = 0
        i += 1
    return cls_labels, reg_labels


def construct_feature(subset, tem_length, feature_size, gt_file, cfg_file, feat_spa_dir, feat_tem_dir, res_dir):
    with open(gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    update_config(cfg_file)
    location_list, duration_list = location_point_back(cfg, to_tensor=False, to_duration=True)

    res_fol_fir = os.path.join(res_dir, subset)
    if not os.path.exists(res_fol_fir):
        os.makedirs(res_fol_fir)

    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] == subset:
            feat_file = os.path.join(feat_spa_dir, 'v_' + vid_name + '.npz')
            if not os.path.exists(feat_file):
                print('feature not exists', vid_name)
                continue

            save_file_name = 'v_' + vid_name + '.npz'
            save_file = os.path.join(res_dir, subset, save_file_name)
            if os.path.exists(save_file):
                print('skip existing file', save_file_name)
                continue

            vid_duration = vid_anns['duration']
            anns = vid_anns['annotations']

            # action [start, end], label
            action_boundary = list()
            for ann in anns:
                # use relative duration
                segment = ann['segment']
                start_loc = (segment[0] / vid_duration) * tem_length
                end_loc = (segment[1] / vid_duration) * tem_length
                data = [start_loc, end_loc]
                action_boundary.append(data)

            # # load temporal feature
            # feat_file = os.path.join(feat_tem_dir, 'v_' + vid_name + '.npz')
            # feat_tem, cnt_tem = load_npz_feat(feat_file)
            # feat_tem = np.transpose(feat_tem)
            # feat_tem = cv2.resize(feat_tem, feature_size, interpolation=cv2.INTER_LINEAR)
            feat_tem = None
            # load spatial feature
            feat_file = os.path.join(feat_spa_dir, 'v_' + vid_name + '.npz')
            feat_spa, cnt = load_npz_feat(feat_file)
            feat_spa = np.transpose(feat_spa)
            feat_spa = cv2.resize(feat_spa, feature_size, interpolation=cv2.INTER_LINEAR)

            # fore-/back- ground label
            cls_label, reg_label = boundary_to_label(action_boundary, location_list, duration_list)

            np.savez(os.path.join(res_dir, save_file),
                     vid_name=vid_name, begin_frame=0, video_length=1,
                     cls_label=cls_label,  reg_label=reg_label,
                     feat_spa=feat_spa, feat_tem=feat_tem)


if __name__ == '__main__':
    args = parse_args()
    construct_feature(args.subset, args.tem_length, args.feature_size, args.gt_file, args.cfg_file, args.feat_spa_dir, args.feat_tem_dir, args.res_dir)

