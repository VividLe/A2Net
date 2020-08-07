import argparse
import os
from zipfile import ZipFile
import shutil
from multiprocessing import Pool


def args_parser():
    parser = argparse.ArgumentParser(description='zip and unzip files')
    parser.add_argument('-ori_dir', default='/disk/yangle/MILESTONES/ActionLocalization/data/thumos/ab_af/test')
    parser.add_argument('-res_dir', default='/disk/yangle/MILESTONES/ActionLocalization/output/datas/test')
    # zip files
    parser.add_argument('-num_feat_files', default=1000)
    parser.add_argument('-tmp_dir', default='/disk/yangle/MILESTONES/ActionLocalization/output/datas/tmp')
    parser.add_argument('-num_proc', default=6)
    # unzip files
    parser.add_argument('-zip_dir', default='/data/yangle/ActionLocalization/data/af_128/zip_feature_tmp')
    args = parser.parse_args()
    return args


def copy_file(name):
    shutil.copyfile(os.path.join(args.ori_dir, name), os.path.join(args.tmp_dir, name))


def zip_files(args):
    file_set = os.listdir(args.ori_dir)
    file_set.sort()
    for i in range(0, len(file_set), args.num_feat_files):
        print(i)
        sel_name_set = file_set[i:i + args.num_feat_files]

        if os.path.exists(args.tmp_dir):
            shutil.rmtree(args.tmp_dir)
        os.makedirs(args.tmp_dir)

        pool = Pool(args.num_proc)
        run = pool.map(copy_file, sel_name_set)
        pool.close()
        pool.join()

        output_file = args.res_dir + '/' + str(i).zfill(5)
        shutil.make_archive(output_file, 'zip', args.tmp_dir)


def unzip_files(args):
    file_set = os.listdir(args.ori_dir)
    file_set = [s for s in file_set if s.endswith('.zip')]
    file_set.sort()

    for name in file_set:
        print(name)
        with ZipFile(os.path.join(args.ori_dir, name), 'r') as zipobj:
            zipobj.extractall(args.res_dir)
        # move zip files
        shutil.move(os.path.join(args.ori_dir, name), os.path.join(args.zip_dir, name))


if __name__ == '__main__':
    global args
    args = args_parser()
    zip_files(args)
