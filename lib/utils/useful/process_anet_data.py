import argparse
import os
import shutil
import cv2
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('-frames_dir', default='/data/home/v-yale/w_MSM/v-yale/ActivityNet/output_sup')
parser.add_argument('-vid_dir', default='/data/home/v-yale/w_MSM/v-yale/ActivityNet/videos_sup')
parser.add_argument('-vid_remain_dir', default='/data/home/v-yale/w_MSM/v-yale/ActivityNet/videos_remain')
parser.add_argument('-log_file', default='/data/home/v-yale/ActionLocalization/lib/utils/sup.log')
parser.add_argument('-match_ratio', default=0.99,
                    help='threshold for judge whether the extract process is finished')
parser.add_argument('-num_worker', default=32, help='parallel worker number')
args = parser.parse_args()


def check_videos(vid_name):
    cap = cv2.VideoCapture(os.path.join(args.vid_dir, vid_name+'.mp4'))
    frame_num_vid = cap.get(7)
    # frame_rate = cap.get(5)

    fol_name = vid_name
    img_dir = os.path.join(args.frames_dir, fol_name)
    if not os.path.exists(img_dir):
        print('skip empty folders %s' % fol_name)
        return
    frame_set = os.listdir(os.path.join(args.frames_dir, fol_name))
    frame_num_ext = len(frame_set)

    # there are 3 file: img_*, flow_x_*, flow_y_*
    num_f, num_m = divmod(frame_num_ext, 3)

    ratio = num_f * 1.0 / frame_num_vid
    if ratio < args.match_ratio:
        print('frame number not match, ratio: %f, video_name: %s' % (ratio, vid_name))


if __name__ == '__main__':
    num_processor = args.num_worker
    fol_set = os.listdir(args.frames_dir)

    pool = Pool(num_processor)
    run = pool.map(check_videos, fol_set)
    pool.close()
    pool.join()
    print('finish data check')


# with open(args.log_file, 'r') as f:
#     lines = f.readlines()
#
# for line in lines:
#     if not "ratio: 0.000" in line:
#         continue
#     datas = line.split(' ')
#     vid_name = datas[-1][:-1]
#     fol_path = os.path.join(args.frames_dir, vid_name)
#     print('remove', fol_path)
#     shutil.rmtree(os.path.join(args.frames_dir, vid_name))
#     shutil.move(os.path.join(args.vid_dir, vid_name+'.mp4'), os.path.join(args.vid_remain_dir, vid_name+'.mp4'))


# fol_rp = '/data/home/v-yale/MSMdata/v-yale/ActivityNet/output_temp'
# vid_ori = '/data/home/v-yale/MSMdata/v-yale/ActivityNet/videos_temp'
# vid_res = '/data/home/v-yale/MSMdata/v-yale/ActivityNet/z_finished'
#
# fol_set = os.listdir(fol_rp)
# fol_set.sort()
#
# for vid_name in fol_set:
#     vid_name = vid_name + '.mp4'
#     print(vid_name)
#     shutil.move(os.path.join(vid_ori, vid_name), os.path.join(vid_res, vid_name))

