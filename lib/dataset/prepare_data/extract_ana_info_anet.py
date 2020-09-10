import argparse
import json
import os
import numpy as np
import pickle
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-subset', default='training')
parser.add_argument('-gt_file', default='/data/home/v-yale/ActionLocalization/lib/dataset/materials/activity_net.v1-3.min.json')
parser.add_argument('-fps', default=25)
parser.add_argument('-diff_duration_th', default=1)
parser.add_argument('-name_idx_file', default='/data/home/v-yale/ActionLocalization/lib/dataset/materials/name_idx_dict.pkl')
parser.add_argument('-tem_feature_dir', default='/data/home/v-yale/w_MSM/v-yale/ActivityNet/cmcs_feat/ANET_FEATURES/anet_i3d_feature_25fps/flow-resize-step16')
parser.add_argument('-res_dir', default='/data/home/v-yale/ActionLocalization/data/anet')
args = parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    frame_cnt = datas['frame_cnt']
    return feature, frame_cnt


if __name__ == '__main__':
    with open(args.gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']
    name_idx = pickle.load(open(args.name_idx_file, 'rb'))

    diff_set = list()
    # collect info
    video_names = list()
    type_v = list()
    type_idx = list()
    start_frame = list()
    end_frame = list()
    frame_num = list()

    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] == args.subset:
            feat_file = os.path.join(args.tem_feature_dir, 'v_' + vid_name + '-flow.npz')
            if not os.path.exists(feat_file):
                # print('feature not exists', vid_name)
                continue

            feat_file = os.path.join(args.tem_feature_dir, 'v_' + vid_name + '-flow.npz')
            feat_tem, cnt_tem = load_npz_feat(feat_file)
            vid_duration = vid_anns['duration']
            anns = vid_anns['annotations']

            # frame difference
            diff = cnt_tem / args.fps - vid_duration
            # the difference no longer than 1 second
            if abs(diff) > args.diff_duration_th:
                print(vid_name, cnt_tem / args.fps, vid_duration)
                # diff_set.append([vid_name, diff])

            for ann in anns:
                # use relative duration
                segment = ann['segment']
                act_end_frame = int(segment[1] * args.fps)
                if (act_end_frame - cnt_tem) > args.fps:
                    print('action out of range, skip', act_end_frame, cnt_tem)
                    continue
                start_frame.append(int(segment[0] * args.fps))
                end_frame.append(act_end_frame)

                video_names.append(vid_name)
                # convert label name to index
                act_name = ann['label']
                type_v.append(act_name)
                type_idx.append(int(name_idx[act_name]))
                frame_num.append(cnt_tem)

    dic_inf = dict()
    dic_inf['video'] = video_names
    dic_inf['type'] = type_v
    dic_inf['type_idx'] = type_idx
    dic_inf['startFrame'] = start_frame
    dic_inf['endFrame'] = end_frame
    dic_inf['frame_num'] = frame_num

    df = pd.DataFrame(dic_inf, columns=['video', 'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)
    res_file = os.path.join(args.res_dir, 'anet_'+args.subset+'.csv')
    df.to_csv(res_file, encoding='utf-8', index=False)
