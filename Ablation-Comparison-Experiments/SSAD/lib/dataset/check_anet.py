import argparse
import json


def args_parser():
    parser = argparse.ArgumentParser(description='show anet annotation information')
    parser.add_argument('-video_name', default='3joaQzU05MY')
    parser.add_argument('-gt_file', default='./materials/activity_net.v1-3.min.json')
    args = parser.parse_args()
    return args


def check_ant(args):

    with open(args.gt_file, 'r') as f:
        gts = json.load(f)
    gt = gts['database']

    vid_anns = gt[args.video_name]
    anns = vid_anns['annotations']
    for ann in anns:
        print(ann['segment'])
        print(ann['label'])

    print('subset: %s, duration: %f' % (vid_anns['subset'], vid_anns['duration']))


if __name__ == '__main__':
    args = args_parser()
    check_ant(args)
