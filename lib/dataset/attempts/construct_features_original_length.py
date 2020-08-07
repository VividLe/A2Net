import argparse
import numpy as np
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('-phase', default='test')
parser.add_argument('-tem_feature_dir', default='/data/home/v-yale/w_MSM/v-yale/BasicDataset/THUMOS/CMCS_feat/THUMOS14_FEATURES/I3D-4/thumos-test-flow-resize')
parser.add_argument('-spa_feature_dir', default='/data/home/v-yale/w_MSM/v-yale/BasicDataset/THUMOS/CMCS_feat/THUMOS14_FEATURES/I3D-4/thumos-test-rgb-resize')
parser.add_argument('-ann_file', default='./pre_process/ann_val.pkl')
parser.add_argument('-class_label', default=[0, 7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97])
parser.add_argument('-vid_names_25fps', default=['video_validation_0000311', 'video_validation_0000420', 'video_validation_0000666', 'video_validation_0000419', 'video_validation_0000484', 'video_validation_0000413'])
parser.add_argument('-res_dir', default='/data/home/v-yale/ActionLocalization/data/thumos/feature_original/test')
args = parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    frame_cnt = datas['frame_cnt']
    return feature, frame_cnt


if __name__ == '__main__':
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    if args.phase == 'val':
        gt_segment = pickle.load(open(args.ann_file, 'rb'))

    file_set = os.listdir(args.tem_feature_dir)
    file_set.sort()
    for file_name in file_set:
        vid_name = file_name[:-9]
    # for vid_name, anns in gt_segment.items():
        print(vid_name)
        save_file = vid_name + '.npz'

        # load feature
        mode = 'flow'
        feat_file = os.path.join(args.tem_feature_dir, vid_name + '-' + mode + '.npz')
        feat_tem, cnt_tem = load_npz_feat(feat_file)
        feat_tem = np.transpose(feat_tem)
        mode = 'rgb'
        feat_file = os.path.join(args.spa_feature_dir, vid_name + '-' + mode + '.npz')
        feat_spa, cnt_spa = load_npz_feat(feat_file)
        feat_spa = np.transpose(feat_spa)
        assert cnt_tem == cnt_spa

        if vid_name in args.vid_names_25fps:
            fps = 25
        else:
            fps = 30
        total_duration = cnt_tem / 30

        # parse gt here
        annotation = list()
        label = list()
        if args.phase == 'val':
            for tmp_ann in anns:
                data = [tmp_ann[0]/total_duration, tmp_ann[1]/total_duration, 1]
                annotation.append(data)
                label.append(args.class_label.index(tmp_ann[2]))
        annotation = np.array(annotation)
        label = np.array(label)

        np.savez(os.path.join(args.res_dir, save_file),
                 vid_name=vid_name, begin_frame=0, video_length=cnt_tem,
                 action=annotation, class_label=label,
                 feat_tem=feat_tem, feat_spa=feat_spa)
