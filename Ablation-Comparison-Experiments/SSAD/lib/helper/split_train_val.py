import os
import random
import shutil

val_per = 0.25
ori_dir = '/data/2/v-yale/ActionLocalization/data/anet_long_short/training'
res_dir = '/data/2/v-yale/ActionLocalization/data/anet_long_short/validation'

if not os.path.exists(res_dir):
    os.makedirs(res_dir)

file_set = os.listdir(ori_dir)
vid_name_set = [s[2:-7] for s in file_set]
vid_names = list(set(vid_name_set))
vid_names.sort()

start_idx = 0.75
end_idx = 1.0
num_vids = len(vid_names)
sel_set = vid_names[round(num_vids * start_idx):round(num_vids * end_idx)]

# sel_set =
print('num:', len(vid_names), len(sel_set))

for name in file_set:
    tmp_vid_name = name[2:-7]
    if tmp_vid_name in sel_set:
        shutil.move(os.path.join(ori_dir, name), os.path.join(res_dir, name))
