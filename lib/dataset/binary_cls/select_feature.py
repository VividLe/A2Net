import argparse
import os
import shutil
import pickle


def args_parser():
    parser = argparse.ArgumentParser(description='select datas for first 40 categories')
    parser.add_argument('-subset', default='training')
    parser.add_argument('-sel_videos_file', default='/data/yangle/ExtractSlowFastFeature/train_val_videos.pkl')
    parser.add_argument('-ori_dir', default='/data/yangle/ActionLocalization/data/af_128')
    parser.add_argument('-res_dir', default='/data/yangle/ActionLocalization/data/af_128_mini')
    args = parser.parse_args()
    return args


def select_datas(args):
    vid_names = pickle.load(open(args.sel_videos_file, 'rb'))
    for name in vid_names:
        file_name = 'v_' + name + '.npz'

        ori_path = os.path.join(args.ori_dir, args.subset, file_name)
        if os.path.exists(ori_path):
            shutil.copy(ori_path, os.path.join(args.res_dir, args.subset, file_name))


if __name__ == '__main__':
    args = args_parser()
    select_datas(args)
