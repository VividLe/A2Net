import argparse
import numpy as np
import pickle
import os
import cv2


def args_parser():
    parser = argparse.ArgumentParser(description='generate sliding window annotations')
    parser.add_argument('-subset', default='training')
    parser.add_argument('-wind_length', default=64)
    parser.add_argument('-feature_size', default=(2048, 64))
    parser.add_argument('-gt_file', default='/disk/yangle/AbHacs/lib/dataset/materials/HACS_segments_v1.1.1.json')
    parser.add_argument('-spa_feature_dir', default='/disk/yangle/AbHacs/dataset/feature/slowfast')
    parser.add_argument('-anns', help='box annotation files, calculate via generate_sliding_window_info.py',
                        default='/disk/yangle/AbHacs/dataset/feature/anns_train.pkl')
    parser.add_argument('-name_idx_dict', help='a dictionary to bridge the category name and index together',
                        default='/disk/yangle/AbHacs/lib/dataset/materials/name_idx_dict.pkl')
    parser.add_argument('-res_dir', default='/disk/yangle/AbHacs/dataset/feature/training')
    args = parser.parse_args()
    return args


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feat_sf']
    return feature


def pad_feature(feature, feat_win):
    t = feature.shape[0]
    feature_slice = feature[-1:, :]
    for _ in range(t, feat_win):
        feature = np.concatenate((feature, feature_slice), axis=0)
    assert feature.shape[0] == feat_win
    return feature


def construct_feature(args, vid_name, feat_file, start_idx, end_idx):
    feature = load_npz_feat(feat_file)
    # ensure the temporal length is no shorter than window_length
    if feature.shape[0] < args.wind_length:
        feature = cv2.resize(feature, args.feature_size, interpolation=cv2.INTER_LINEAR)
    tem_length = feature.shape[0]

    end_tmp = min(end_idx, feature.shape[0])
    feat_sel = feature[start_idx:end_tmp, :]

    if end_idx >= feature.shape[0] + 1:
        print('%s padding feature, original length: %d, desired length: %d' % (vid_name, feature.shape[0], end_idx))
        feat_sel = pad_feature(feat_sel, feat_win=args.wind_length)
    feat = np.transpose(feat_sel)
    return feat, tem_length


def construct_datas(args):
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    with open(args.anns, 'rb') as f:
        anns = pickle.load(f)

    with open(args.name_idx_dict, 'rb') as f:
        name_idx_dict = pickle.load(f)

    for vid_name, vid_anns in anns.items():
        print(vid_name)
        gt_windows = vid_anns[0]
        gt_labels = vid_anns[1]
        gt_boxs = vid_anns[2]

        for gt_window, gt_label, gt_box in zip(gt_windows, gt_labels, gt_boxs):

            start_idx = int(gt_window[0])
            end_idx = int(gt_window[1])

            save_file = 'v_' + vid_name + '_' + str(start_idx).zfill(4) + '.npz'

            feat_file = os.path.join(args.spa_feature_dir, 'v_' + vid_name + '.npz')
            feat_spa, tem_length = construct_feature(args, vid_name, feat_file, start_idx, end_idx)

            # todo: class_label
            np.savez(os.path.join(args.res_dir, save_file),
                     vid_name=vid_name, begin_idx=start_idx, tem_length=tem_length,
                     action=gt_box, class_label=name_idx_dict[gt_labels[0]], feat_spa=feat_spa)
  

if __name__ == '__main__':
    args = args_parser()
    construct_datas(args)
