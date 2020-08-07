import argparse
import json
import os
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-subset', default='training')
parser.add_argument('-gt_file', default='./materials/activity_net.v1-3.min.json')
parser.add_argument('-name_idx_file', default='./materials/name_idx_dict.pkl')
parser.add_argument('-tem_feature_dir', default='=/hdfs/nextmsra/v-yale/Basicdata/activitynet/cmcs_feat/flow-resize-step16')
parser.add_argument('-spa_feature_dir', default='/hdfs/nextmsra/v-yale/Basicdata/activitynet/cmcs_feat/rgb-resize-step16')
parser.add_argument('-diff_duration_th', default=1)
parser.add_argument('-window_width', default=2048)
parser.add_argument('-fps', default=25)
parser.add_argument('-feat_sample_interval', default=16)
parser.add_argument('-res_dir', default='/hdfs/nextmsra/v-yale/SSAD_anet/SSAD_wind/data/anet/feature')
args = parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    frame_cnt = datas['frame_cnt']
    return feature, frame_cnt


def sliding_window(window_width, frame_count):
    stride = window_width / 4
    num_winds = int((frame_count + stride - window_width) / stride)
    winds_start = [i * stride for i in range(num_winds)]
    last_wind_start = frame_count - window_width
    if last_wind_start not in winds_start:
        winds_start.append(last_wind_start)
    return winds_start


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
            feat_file = os.path.join(args.tem_feature_dir, 'v_' + vid_name + '-flow.npz')
            if not os.path.exists(feat_file):
                # print('feature not exists', vid_name)
                continue

            # load feature
            mode = 'flow'
            feat_file = os.path.join(args.tem_feature_dir, 'v_' + vid_name + '-' + mode + '.npz')
            feat_tem, cnt_tem = load_npz_feat(feat_file)
            if cnt_tem <= args.window_width:
                print('short videos, skip')
                continue
            feat_tem = np.transpose(feat_tem)
            mode = 'rgb'
            feat_file = os.path.join(args.spa_feature_dir, 'v_' + vid_name + '-' + mode + '.npz')
            feat_spa, cnt_spa = load_npz_feat(feat_file)
            feat_spa = np.transpose(feat_spa)
            assert cnt_tem == cnt_spa

            vid_duration = vid_anns['duration']
            anns = vid_anns['annotations']

            winds_start = sliding_window(args.window_width, cnt_tem)
            for iord, start_frame in enumerate(winds_start):
                save_file_name = 'v_' + vid_name + str(iord).zfill(2) + '.npz'
                save_file = os.path.join(args.res_dir, args.subset, save_file_name)

                # action [start, end], label
                annotations = list()
                labels = list()
                if args.subset == 'training':
                    for ann in anns:
                        # use relative duration
                        segment = ann['segment']
                        act_start = segment[0] * args.fps
                        act_end = segment[1] * args.fps
                        if (act_start < start_frame) or (act_start > start_frame+args.window_width) or (act_end < start_frame) or (act_end > start_frame+args.window_width):
                            print('out of window, skip')
                            continue
                        data = [(act_start - start_frame) / args.window_width, (act_end - start_frame) / args.window_width]
                        annotations.append(data)
                        # convert label name to index
                        act_name = ann['label']
                        label = int(name_idx[act_name])
                        labels.append(label)

                    if len(annotations) > 0:
                        annotations = np.array(annotations)
                        labels = np.array(labels)

                start_idx = int(start_frame / args.feat_sample_interval)
                end_idx = int(start_idx + args.window_width / args.feat_sample_interval)
                tmp_feat_tem = feat_tem[:, start_idx:end_idx]
                tmp_feat_spa = feat_spa[:, start_idx:end_idx]

                np.savez(os.path.join(args.res_dir, save_file),
                         vid_name=vid_name, begin_frame=0, video_length=args.window_width,
                         action=annotations, class_label=labels,
                         feat_tem=tmp_feat_tem, feat_spa=tmp_feat_spa)

