import argparse
import json
import os
import pickle
import moviepy.editor as mpy
from multiprocessing import Pool
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-subset', default='training')
parser.add_argument('-ann_file', default='../materials/activity_net.v1-3.min.json')
parser.add_argument('-videos_dir', default='/data/guoxi/ActLoc/dataset/ActivityNet/videos')
parser.add_argument('-save_cls_dir', default='/data/guoxi/ActLoc/ActionLocalization/data/binary_cls/cls_videos_fg')
parser.add_argument('-num_processor', default=32)
parser.add_argument('-idx_name_file', default='../materials/idx_name_dict.pkl')
args = parser.parse_args()


def extract_anno(vid_name):
    vid_anns = gt[vid_name]
    anns = vid_anns['annotations']
    vid_dur = vid_anns['duration']
    for order, ann in enumerate(anns):
        segment = ann['segment']

        start_time = segment[0]
        start_min, start_sec = divmod(start_time, 60)
        end_time = min(segment[1], vid_dur)
        end_min, end_sec = divmod(end_time, 60)
        duration = end_time - start_time
        # filter out short actions
        if duration < 1:
            return

        order = order + 1
        res_name = 'v_' + vid_name + '_' + str(order) + '.mp4'
        res_file = os.path.join(save_dir, res_name)
        if os.path.exists(res_file):
            return
        print('res_file:', res_file)

        video_file = os.path.join(args.videos_dir, 'v_' + vid_name + '.mp4')
        if not os.path.exists(video_file):
            return
        clip = mpy.VideoFileClip(video_file)
        content = clip.subclip((start_min, start_sec), (end_min, end_sec))
        content.write_videofile(res_file)


if __name__ == '__main__':

    with open(args.ann_file, 'r') as f:
        gts = json.load(f)
    global gt
    gt = gts['database']

    idx_name_dict = pickle.load(open(args.idx_name_file, 'rb'))

    for i in range(1, 41):
        global cls_name
        cls_name = idx_name_dict[i]
        datas = cls_name.split(' ')
        cls_name_seq = datas[0]
        for d in datas[1:]:
            cls_name_seq = cls_name_seq + '_' + d

        global save_dir
        save_dir = os.path.join(args.save_cls_dir, cls_name_seq)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # collect target videos
        vid_name_sels = list()
        for vid_name, vid_anns in gt.items():
            if vid_anns['subset'] != args.subset:
                continue
            anns = vid_anns['annotations']
            label = anns[0]['label']
            if label != cls_name:
                continue
            vid_name_sels.append(vid_name)

        print('sel numbers', len(vid_name_sels))
        # calculate
        pool = Pool(args.num_processor)
        run = pool.map(extract_anno, vid_name_sels)
        pool.close()
        pool.join()
