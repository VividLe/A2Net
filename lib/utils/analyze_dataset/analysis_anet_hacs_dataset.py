import argparse
import json
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt


def args_parser():
    parser = argparse.ArgumentParser(description='analyze ActivityNet or HACS annotation infomation')
    parser.add_argument('-ann_file', default='../HACS_v1.1.1/HACS_segments_v1.1.1.json')
    parser.add_argument('-subset', default='training')
    parser.add_argument('-bin_number', default=200)
    parser.add_argument('-num_show_ann', default=500)
    parser.add_argument('-pixel_per_s', default=100)
    parser.add_argument('-col', default=100)
    parser.add_argument('-fore_RGB', default=[255, 0, 0])
    parser.add_argument('-bg_RGB', default=[0, 255, 0])
    parser.add_argument('-res_dir', default='./vis_actions')
    args = parser.parse_args()
    return args


def stac_action_durs(args):
    with open(args.ann_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    duration_list = list()
    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] != args.subset:
            continue

        vid_duration = float(vid_anns['duration'])
        anns = vid_anns['annotations']
        for ann in anns:
            segment = ann['segment']
            # duration = segment[1] - segment[0]
            duration_r = (segment[1] - segment[0]) / vid_duration
            duration_list.append(duration_r)

    duration_list.sort()
    print('action instance number %d' % len(duration_list))

    # # show relative duration
    # stride = int(len(duration_list) / 20)
    # for i in range(1, len(duration_list), stride):
    #     print(duration_list[i])

    # # draw hist figure
    # plt.hist(duration_list[:-5], bins=args.bin_number)
    # plt.show()


def draw_ann_img(vid_anns):
    vid_duration = float(vid_anns['duration'])
    img = np.zeros((args.col, int(vid_duration * args.pixel_per_s), 3))
    img[:, :] = args.bg_RGB

    anns = vid_anns['annotations']
    for ann in anns:
        segment = ann['segment']
        act_start_pixel = int(segment[0] * args.pixel_per_s)
        act_end_pixel = int(segment[1] * args.pixel_per_s)
        img[:, act_start_pixel:act_end_pixel] = args.fore_RGB

    return img


def show_ann(args):
    with open(args.ann_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    vid_names = list()
    for vid_name, vid_anns in gt.items():
        if vid_anns['subset'] != args.subset:
            continue
        vid_names.append(vid_name)

    sel_names = random.sample(vid_names, args.num_show_ann)
    for vid_name in sel_names:
        vid_anns = gt[vid_name]
        # save visualization image
        img = draw_ann_img(vid_anns)
        rendering_img = Image.fromarray(img.astype('uint8')).convert('RGB')
        rendering_img_name = vid_name + '.jpg'
        img_save_path = os.path.join(args.res_dir, rendering_img_name)
        rendering_img.save(img_save_path)


if __name__ == '__main__':
    args = args_parser()
    # stac_action_durs(args)
    show_ann(args)
