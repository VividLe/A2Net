import argparse
import json
import os
import numpy as np
import pickle
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subset', default='validation')
    parser.add_argument('-max_length', default=2048)
    parser.add_argument('-feature_size', default=(2048, 1024))
    parser.add_argument('-gt_file', default='../materials/activity_net.v1-3.min.json')
    parser.add_argument('-layer_length_file', default='/data/yangle/ActionLocalization/lib/models/num_each_layer.pkl')
    parser.add_argument('-act_ranges', default=[[0, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024], [1024, 2048]])
    parser.add_argument('-loc_file', default='/data/yangle/ActionLocalization/lib/models/center_location.pkl')
    parser.add_argument('-feat_spa_dir', default='/data/yangle/dataset/ActivityNet/feature/I3D_stride4_filter/rgb')
    parser.add_argument('-feat_tem_dir', default='/data/yangle/dataset/ActivityNet/feature/I3D_stride4_filter/flow')
    parser.add_argument('-res_dir', default='/data/yangle/ActionLocalization/data/olength_flipped')
    return parser.parse_args()


def load_npz_feat(file):
    datas = np.load(open(file, 'rb'))
    feature = datas['feature']
    feature = feature[0, :, :]
    return feature


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


def gene_cls_label(action_boundary, layers_len, act_ranges):
    cls_label_list = list()
    reg_label_list = list()

    tem_length = layers_len[0]
    for layer_len, act_range in zip(layers_len, act_ranges):
        # TODO: potential bug when the temporal length is odd. [the last value is zero]
        cls_label = np.zeros(layer_len, dtype=np.int)
        reg_label = np.zeros((layer_len, 2), dtype=np.float)

        start_idx = int(act_range[0] / 2)
        stride = act_range[0]
        if stride == 0:
            stride = 1

        # enumerate each temporal location
        for i, loc in enumerate(list(range(start_idx, tem_length, stride))):
            flag, reg_left, reg_right = idx_in_actions(loc, act_range, action_boundary)
            if flag:
                cls_label[i] = 1
                reg_label[i, 0] = reg_left
                reg_label[i, 1] = reg_right
        cls_label_list.append(cls_label)
        reg_label_list.append(reg_label)

    # collect result
    cls_labels = np.concatenate(cls_label_list, axis=0)
    reg_labels = np.concatenate(reg_label_list, axis=0)
    return cls_labels, reg_labels


def gene_cls_reg_labels(is_flip, anns, vid_duration, tem_length, layer_num_dict):
    action_boundary = list()
    for ann in anns:
        # use relative duration
        segment = ann['segment']
        start_time = segment[0]
        end_time = segment[1]
        if is_flip:
            tmp = vid_duration - end_time
            end_time = vid_duration - start_time
            start_time = tmp
        start_loc = (start_time / vid_duration) * tem_length
        end_loc = (end_time / vid_duration) * tem_length
        data = [start_loc, end_loc]
        action_boundary.append(data)

    # fore-/back- ground label
    layers_len = layer_num_dict[tem_length]
    num_layer = len(layers_len)
    cls_label, reg_label = gene_cls_label(action_boundary, layers_len, args.act_ranges[:num_layer])
    assert cls_label.max() == 1
    return cls_label, reg_label



def construct_feature(subset, gt_file, feat_spa_dir, feat_tem_dir, res_dir):
    with open(gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    layer_num_dict = pickle.load(open(args.layer_length_file, 'rb'))
    tem_locations_dict = pickle.load(open(args.loc_file, 'rb'))

    res_fol_dir = os.path.join(res_dir, subset)
    if not os.path.exists(res_fol_dir):
        os.makedirs(res_fol_dir)

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

            # load temporal feature
            feat_file = os.path.join(feat_tem_dir, 'v_' + vid_name + '.npz')
            feat_tem = load_npz_feat(feat_file)
            feat_tem = np.transpose(feat_tem)
            # load spatial feature
            feat_file = os.path.join(feat_spa_dir, 'v_' + vid_name + '.npz')
            feat_spa = load_npz_feat(feat_file)
            feat_spa = np.transpose(feat_spa)
            # ensure the same temporal length
            tem_length = min(feat_tem.shape[1], feat_spa.shape[1])
            feat_tem = feat_tem[:, :tem_length]
            feat_spa = feat_spa[:, :tem_length]

            # for features with temporal length longer than 2048, we scale it to 2048
            if tem_length >= args.max_length:
                tem_length = args.max_length
                feat_spa = cv2.resize(feat_spa, args.feature_size, interpolation=cv2.INTER_LINEAR)
                feat_tem = cv2.resize(feat_tem, args.feature_size, interpolation=cv2.INTER_LINEAR)

            # center location
            location = tem_locations_dict[tem_length]

            vid_duration = vid_anns['duration']
            anns = vid_anns['annotations']

            # original order
            cls_label, reg_label = gene_cls_reg_labels(False, anns, vid_duration, tem_length, layer_num_dict)
            feat_spas = [feat_spa]
            feat_tems = [feat_tem]
            cls_labels = [cls_label]
            reg_labels = [reg_label]
            cls_label, reg_label = gene_cls_reg_labels(True, anns, vid_duration, tem_length, layer_num_dict)
            feat_spas.append(feat_spa[:, ::-1])
            feat_tems.append(feat_tem[:, ::-1])
            cls_labels.append(cls_label)
            reg_labels.append(reg_label)

            # # action [start, end], label
            # action_boundary = list()
            # for ann in anns:
            #     # use relative duration
            #     segment = ann['segment']
            #     start_loc = (segment[0] / vid_duration) * tem_length
            #     end_loc = (segment[1] / vid_duration) * tem_length
            #     data = [start_loc, end_loc]
            #     action_boundary.append(data)
            #
            # # fore-/back- ground label
            # layers_len = layer_num_dict[tem_length]
            # num_layer = len(layers_len)
            # cls_label, reg_label = gene_cls_label(action_boundary, layers_len, args.act_ranges[:num_layer])
            # assert cls_label.max() == 1

            np.savez(os.path.join(res_dir, save_file),
                     vid_name=vid_name, begin_frame=0, tem_length=tem_length, location=location,
                     cls_label=cls_labels,  reg_label=reg_labels, feat_spa=feat_spas, feat_tem=feat_tems)


if __name__ == '__main__':
    args = parse_args()
    # imp(args)
    construct_feature(args.subset, args.gt_file, args.feat_spa_dir, args.feat_tem_dir, args.res_dir)

