import argparse
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-phase', default='test')
parser.add_argument('-tem_feature_dir', default='/data/home/v-yale/w_MSM/v-yale/BasicDataset/THUMOS/CMCS_feat/THUMOS14_FEATURES/I3D-4/thumos-test-flow-resize')
parser.add_argument('-spa_feature_dir', default='/data/home/v-yale/w_MSM/v-yale/BasicDataset/THUMOS/CMCS_feat/THUMOS14_FEATURES/I3D-4/thumos-test-rgb-resize')
parser.add_argument('-wind_info_file', default='/data/home/v-yale/ActionLocalization/data/thumos/test/window_info.log')
parser.add_argument('-gt_box_file', default='/data/home/v-yale/ActionLocalization/data/thumos/test/gt_box.pkl')
parser.add_argument('-gt_label_file', default='/data/home/v-yale/ActionLocalization/data/thumos/test/gt_label.pkl')
parser.add_argument('-res_dir', default='/data/home/v-yale/ActionLocalization/data/thumos/feature/test')
parser.add_argument('-sample_interval', default=4)
parser.add_argument('-wind_length', default=128)
args = parser.parse_args()


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


if __name__ == '__main__':
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    with open(args.wind_info_file, 'r') as f:
        lines = f.readlines()
    if args.phase == 'val':
        gt_box = pickle.load(open(args.gt_box_file, 'rb'))
        gt_label = pickle.load(open(args.gt_label_file, 'rb'))

    for iord, line in enumerate(lines):
        begin_frame, vid_name = line.split(',')
        vid_name = vid_name[1:-1]
        save_file = vid_name + '_' + begin_frame.zfill(5) + '.npz'

        if args.phase == 'val':
            info = gt_box[iord]
            label = gt_label[iord]
        else:
            info = None
            label = None

        start_idx = int(int(begin_frame) / args.sample_interval)
        end_idx = start_idx + args.wind_length

        mode = 'flow'
        feat_file = os.path.join(args.tem_feature_dir, vid_name + '-' + mode + '.npz')
        feat_tem = construct_feature(vid_name, feat_file, start_idx, end_idx)

        mode = 'rgb'
        feat_file = os.path.join(args.spa_feature_dir, vid_name + '-' + mode + '.npz')
        feat_spa = construct_feature(vid_name, feat_file, start_idx, end_idx)

        if os.path.exists(os.path.join(args.res_dir, save_file)):
            print('already exists', line)

        np.savez(os.path.join(args.res_dir, save_file),
                 vid_name=vid_name, begin_frame=int(begin_frame),
                 action=info, class_label=label,
                 feat_tem=feat_tem, feat_spa=feat_spa)
