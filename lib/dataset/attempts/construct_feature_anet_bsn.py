import argparse
import json
import os
import numpy as np
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-subset', default='training')
parser.add_argument('-gt_file', default='./materials/activity_net.v1-3.min.json')
parser.add_argument('-name_idx_file', default='./materials/name_idx_dict.pkl')
parser.add_argument('-frame_rate', default=25)
parser.add_argument('-bsn_feature_dir', default='/data/home/v-yale/BSN_pt/data/activitynet_feature_cuhk/csv_mean_100')
parser.add_argument('-res_dir', default='/data/home/v-yale/ActionLocalization/data/anet/feature_bsn')
args = parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    frame_cnt = datas['frame_cnt']
    return feature, frame_cnt


def load_csv_feat(file):
    datas = pd.read_csv(file)
    feature = datas.values[:, :]  # [100, 400]
    feature = np.transpose(feature)
    return feature


if __name__ == '__main__':
    with open(args.gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']
    name_idx = pickle.load(open(args.name_idx_file, 'rb'))
    res_fol_fir = os.path.join(args.res_dir, args.subset)
    if not os.path.exists(res_fol_fir):
        os.makedirs(res_fol_fir)

    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] == args.subset:
            feat_file = os.path.join(args.bsn_feature_dir, 'v_' + vid_name + '.csv')
            if not os.path.exists(feat_file):
                print('feature not exists', vid_name)
                continue

            save_file_name = 'v_' + vid_name + '.npz'
            save_file = os.path.join(args.res_dir, args.subset, save_file_name)
            # if os.path.exists(save_file):
            #     print('skip existing file', save_file_name)
            #     continue

            vid_duration = vid_anns['duration']
            video_len = vid_duration * args.frame_rate
            anns = vid_anns['annotations']

            # action [start, end], label
            annotations = list()
            labels = list()
            for ann in anns:
                # use relative duration
                segment = ann['segment']
                data = [segment[0] / vid_duration, segment[1] / vid_duration, 1]
                annotations.append(data)
                # convert label name to index
                act_name = ann['label']
                label = int(name_idx[act_name])
                labels.append(label)
            annotations = np.array(annotations)
            labels = np.array(labels)
            # print(vid_name, annotations, labels)

            # load feature
            feat_file = os.path.join(args.bsn_feature_dir, 'v_'+vid_name + '.csv')
            feature = load_csv_feat(feat_file)

            np.savez(os.path.join(args.res_dir, save_file),
                     vid_name=vid_name, begin_frame=0, video_length=video_len,
                     action=annotations, class_label=labels,
                     feat_tem=feature, feat_spa=feature)


