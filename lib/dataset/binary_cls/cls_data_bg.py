import argparse
import json
import os
import pickle
import moviepy.editor as mpy
from multiprocessing import Pool
import numpy as np
import random


parser = argparse.ArgumentParser()
parser.add_argument('-subset', default='training')
parser.add_argument('-ann_file', default='../materials/activity_net.v1-3.min.json')
parser.add_argument('-videos_dir', default='/data/guoxi/ActLoc/dataset/ActivityNet/videos')
parser.add_argument('-save_cls_dir', default='/data/guoxi/ActLoc/ActionLocalization/data/binary_cls/cls_videos_bg')
parser.add_argument('-num_processor', default=32)
parser.add_argument('-idx_name_file', default='../materials/idx_name_dict.pkl')
parser.add_argument('-mini_act_dur', default=1)
parser.add_argument('-bg_fg_ratio', default=3)
args = parser.parse_args()


def valid_bg(fg_anns, vid_dur):
    bgs = list()
    bgs.append([0, fg_anns[0][0]])
    for i in range(len(fg_anns)-1):
        tmp = [fg_anns[i][1], fg_anns[i+1][0]]
        bgs.append(tmp)
    bgs.append([fg_anns[-1][1], vid_dur])
    bgs = [bg for bg in bgs if bg[1]-bg[0] > args.mini_act_dur]
    return bgs


def generate_range():
    while True:
        d1 = random.random()
        d2 = random.random()
        if abs(d2 - d1) > 0.1:
            break
    if d1 < d2:
        return [d1, d2]
    else:
        return [d2, d1]


def adjust_frag(frag, r):
    act_dur = frag[1] - frag[0]
    start = frag[0] + act_dur * r[0]
    end = frag[0] + act_dur * r[1]
    if end - start > args.mini_act_dur:
        return [start, end]
    else:
        return frag


def extract_anno(vid_name):
    vid_anns = gt[vid_name]
    anns = vid_anns['annotations']
    vid_dur = vid_anns['duration']

    fg_anns = [ann['segment'] for ann in anns]
    bgs_ori = valid_bg(fg_anns, vid_dur)
    if len(bgs_ori) == 0:
        return

    # num_bg = len(fg_anns) * args.bg_fg_ratio
    # bgs_sel = list()
    # for _ in range(num_bg):
    #     fragment = random.choice(bgs_ori)
    #     frag_range = generate_range()
    #     frag_tmp = adjust_frag(fragment, frag_range)
    #     bgs_sel.append(frag_tmp)

    for order, ann in enumerate(bgs_ori):
        start_time = ann[0]
        start_min, start_sec = divmod(start_time, 60)
        end_time = ann[1]
        end_min, end_sec = divmod(end_time, 60)
        duration = end_time - start_time
        # filter out short actions
        if duration < 1:
            continue

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
