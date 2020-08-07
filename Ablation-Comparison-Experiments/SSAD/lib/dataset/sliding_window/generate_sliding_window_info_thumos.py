import os
import pickle
import numpy as np
import pandas as pd
import argparse


def window_data(start_frame, ann_df, video_name):
    # TODO: parameter
    window_size = 512.0
    end_frame = start_frame + window_size

    label = list()
    box = list()
    window = [int(start_frame), video_name]
    # TODO: parameter
    class_label = [7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
    class_label = [0] + class_label

    for i in range(len(ann_df)):
        act_start = ann_df.startFrame.values[i]
        act_end = ann_df.endFrame.values[i]
        assert act_end > act_start
        overlap = min(end_frame, act_end) - max(start_frame, act_start)
        overlap_ration = overlap * 1.0 / (act_end - act_start)
        # print('overlap_ration', overlap_ration)

        # TODO: parameter
        overlap_ratio_threshold = 0.9
        if overlap_ration > overlap_ratio_threshold:
            gt_start = max(start_frame, act_start) - start_frame
            gt_end = min(end_frame, act_end) - start_frame
            #
            # num_classes = 21
            # one_hot = [0] * num_classes
            # one_hot[class_label.index(ann_df.type_idx.values[i])] = 1
            # label.append(one_hot)
            label.append(class_label.index(ann_df.type_idx.values[i]))
            box.append([gt_start*1.0/window_size, gt_end*1.0/window_size, overlap_ration])

    box = np.array(box).astype('float32')
    label = np.array(label)
    return label, box, window


def sliding_window(ann_df, video_name, is_train=True):
    # TODO: use config
    window_size = 512.0
    video_ann_df = ann_df[ann_df.video == video_name]
    frame_count = video_ann_df.frame_num.values[0]
    if is_train:
        stride = window_size / 4
    else:
        stride = window_size / 2

    num_window = int(1.0*(frame_count + stride - window_size) / stride)
    windows_start = [i * stride for i in range(num_window)]
    if is_train and num_window == 0:
        windows_start = [0]
    if frame_count > window_size:
        windows_start.append(frame_count - window_size)

    label = list()
    box = list()
    window = list()
    for start in windows_start:
        if is_train:
            label_tmp, box_tmp, window_tmp = window_data(start, video_ann_df, video_name)
            if len(label_tmp) > 0:
                label.append(label_tmp)
                box.append(box_tmp)
                window.append(window_tmp)
        else:
            window.append([int(start), video_name])
    return label, box, window


def video_process(ann_df, is_train=True):
    # the video_name_list contain 200/213 different video names, so we need list(set()) operation
    video_name_list = list(set(ann_df.video.values[:].tolist()))
    # we should make sure the generated file in the same order
    video_name_list.sort()
    label = list()
    boxes = list()
    window = list()

    for video_name in video_name_list:
        label_tmp, box_tmp, window_tmp = sliding_window(ann_df, video_name, is_train)
        if is_train and (len(label_tmp) > 0):
            label.extend(label_tmp)
            boxes.extend(box_tmp)
        window.extend(window_tmp)
    return label, boxes, window


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ann_path', default='/data/home/v-yale/ActionLocalization/data/thumos/test/thumos14_test_annotation.csv')
    parser.add_argument('-info_dir', default='/data/home/v-yale/ActionLocalization/data/thumos/test')
    args = parser.parse_args()

    ann_df = pd.read_csv(args.ann_path)
    gt_label, gt_box, gt_windows = video_process(ann_df, is_train=False)
    # with open(os.path.join(args.info_dir, 'gt_label.pkl'), 'wb') as f:
    #     pickle.dump(gt_label, f)
    # with open(os.path.join(args.info_dir, 'gt_box.pkl'), 'wb') as f:
    #     pickle.dump(gt_box, f)
    # todo: I can store this in .pkl file
    with open(os.path.join(args.info_dir, 'window_info.log'), 'w') as f:
        f.writelines("%d, %s\n" % (gt_window[0], gt_window[1]) for gt_window in gt_windows)
