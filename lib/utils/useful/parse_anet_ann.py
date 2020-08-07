import argparse
import json
import os
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-gt_file', default='../activity_net.v1-3.min.json')
parser.add_argument('-name_file', default='../test.pkl',
                    help='file to save extracted video names')
parser.add_argument('-vid_dir', default='/data/home/v-yale/MSMdata/v-yale/ActivityNet/videos',
                    help='directory to *.mp4 videos')
parser.add_argument('-available_set', default=['training', 'validation', 'testing'])
args = parser.parse_args()


with open(args.gt_file, 'r') as fobj:
    ann = json.load(fobj)
anns = ann['database']

name_list = list()
for vid_name, vid_ann in anns.items():
    if vid_ann['subset'] in args.available_set:
        vid_file = os.path.join(args.vid_dir, 'v_'+vid_name+'.mp4')
        if os.path.exists(vid_file):
            name_list.append(vid_name)
        else:
            print('not exists', vid_file)

print('processed videos number %d' % len(name_list))
pickle.dump(name_list, open(args.name_file, 'wb'))
