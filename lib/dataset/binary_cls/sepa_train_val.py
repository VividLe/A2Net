import argparse
import os
import shutil
import json


def args_parser():
    parser = argparse.ArgumentParser(description='separate train val data')
    parser.add_argument('-ann_file', default='../materials/activity_net.v1-3.min.json')
    parser.add_argument('-ori_dir', default='/data/guoxi/ActLoc/ActionLocalization/data/binary_cls/cls_videos_fg')
    parser.add_argument('-train_dir', default='/data/guoxi/ActLoc/ActionLocalization/data/binary_cls/training/action')
    parser.add_argument('-val_dir', default='/data/guoxi/ActLoc/ActionLocalization/data/binary_cls/validation/action')
    args = parser.parse_args()
    return args


def sepa_train_val(args):
    # gt info
    gts = json.load(open(args.ann_file, 'r'))
    gt = gts['database']

    fol_set = os.listdir(args.ori_dir)
    fol_set.sort()

    for fol_name in fol_set:
        vid_dir = os.path.join(args.ori_dir, fol_name)
        vid_set = os.listdir(vid_dir)
        vid_set.sort()

        for vid_name in vid_set:
            name = vid_name[2:-6]
            # dispose videos with multiple actions
            if name[-1] == '_':
                name = name[:-1]

            ann = gt[name]
            if ann['subset'] == 'training':
                res_dir = args.train_dir
            elif ann['subset'] == 'validation':
                res_dir = args.val_dir
            else:
                assert "subset error"

            # print(os.path.join(vid_dir, vid_name), os.path.join(res_dir, vid_name))
            shutil.move(os.path.join(vid_dir, vid_name), os.path.join(res_dir, vid_name))


if __name__ == '__main__':
    args = args_parser()
    sepa_train_val(args)
