import argparse
import os
import shutil


def copy_result(args):
    # fol_set = os.listdir(args.fol_dir)
    # fol_set.sort()
    fol_set = ['ori', 'lr_step']

    for fol_name in fol_set:
        ori_dir = os.path.join(args.fol_dir, fol_name)
        file_set = os.listdir(ori_dir)
        json_file_set = [s for s in file_set if s.endswith('.txt')]

        for name in json_file_set:
            print(os.path.join(args.res_dir, fol_name+name))
            shutil.copy(os.path.join(ori_dir, name), os.path.join(args.res_dir, fol_name+name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='collect result')
    parser.add_argument('-fol_dir', default='/disk/yangle/ActLocSSAD_complete/output')
    parser.add_argument('-res_dir', default='/disk/yangle/ActLocSSAD_complete/output/predictions')
    args = parser.parse_args()
    copy_result(args)

