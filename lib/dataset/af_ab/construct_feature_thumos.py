import argparse
import numpy as np
import os
import pickle
import sys

sys.path.append('/disk/peiliang/ab_af_thumos/lib')

from config import cfg, update_config
from core.prediction_box_match import location_point_back


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-phase', default='val')
    # parser.add_argument('-sample_interval', default=4)
    parser.add_argument('-wind_length', default=64)
    parser.add_argument('--data', default='/disk/peiliang/ActLocSSAD_complete/data/thumos/feature/')
    parser.add_argument('-cfg_file', default='/disk/peiliang/ab_af_thumos/experiments/thumos/ab_af_thumos.yaml')
    parser.add_argument('-res_feat_dir', help='directory to constructed feature',
                        default='/disk/peiliang/ab_af_thumos/data/af/val')
    args = parser.parse_args()

    return args


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    return feature


def pad_feature(feature, feat_win=128):
    t = feature.shape[0]
    feature_slice = feature[-1:, :]
    for _ in range(t, feat_win):
        feature = np.concatenate((feature, feature_slice), axis=0)
    assert feature.shape[0] == feat_win
    return feature


def construct_feature(vid_name, feat_file, start_idx, end_idx):
    feature = load_npz_feat(feat_file)
    end_tmp = min(end_idx, feature.shape[0])
    feat_sel = feature[start_idx:end_tmp, :]

    if end_idx >= feature.shape[0] + 1:
        print('%s padding feature, original length: %d, desired length: %d' % (vid_name, feature.shape[0], end_idx))
        feat_sel = pad_feature(feat_sel, feat_win=args.wind_length)
    feat = np.transpose(feat_sel)
    return feat


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


if __name__ == '__main__':
    args = args_parser()
    update_config(args.cfg_file)
    location_list, duration_list = location_point_back(cfg, to_tensor=False, to_duration=True)

    # for loc, dur in zip(location_list, duration_list):
    #     print(loc, dur)

    if not os.path.exists(args.res_feat_dir):
        os.makedirs(args.res_feat_dir)

    data_dir = os.path.join(args.data, args.phase)
    data_names = os.listdir(data_dir)

    for data_name in data_names:
        data_file = os.path.join(data_dir, data_name)
        data = np.load(data_file)
        begin_frame = data['begin_frame']
        vid_name = str(data['vid_name'])
        save_file = vid_name + '_' + str(begin_frame).zfill(5) + '.npz'

        if args.phase == 'val':

            cate_data = data['class_label']
            info = data['action']
            num_box = info.shape[0]
            action_boundary = list()
            for iact in range(num_box):
                action = info[iact, :]
                start_loc = action[0] * args.wind_length
                end_loc = action[1] * args.wind_length
                act_tmp = [start_loc, end_loc]
                action_boundary.append(act_tmp)
            # fore-/back- ground label
            cls_label, reg_label = boundary_to_label(action_boundary, location_list, duration_list)

        else:
            info = None
            cls_label = None
            reg_label = None
            cate_data = None

        feat_tem = data['feat_tem']
        feat_spa = data['feat_spa']
        if os.path.exists(os.path.join(args.res_feat_dir, save_file)):
            print('already exists', data_name)

        np.savez(os.path.join(args.res_feat_dir, save_file),
                 vid_name=vid_name, begin_frame=int(begin_frame),
                 cls_label=cls_label, reg_label=reg_label, action=info, cate_label=cate_data,
                 feat_tem=feat_tem, feat_spa=feat_spa)

