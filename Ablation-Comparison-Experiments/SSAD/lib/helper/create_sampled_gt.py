import json
import os

# feat_dir = '/data/2/v-yale/ActionLocalization/data/anet_long_short/split1/merge_ori'
# ori_gt_file = '/data/2/v-yale/ActionLocalization/data/anet_10s/mini_gt.json'
#
# gts = json.load(open(ori_gt_file, 'r'))
# gt = gts['database']  # 3156
#
# feat_set = os.listdir(feat_dir)
# feat_set = [s[2:-7] for s in feat_set]
# feat_set = list(set(feat_set))  # 2444
#
# for vid_name, vid_anns in gt.items():
#     if vid_name not in feat_set:
#         print(vid_anns['subset'], vid_name,)

ori_gt_file = '/data/2/v-yale/ActionLocalization/data/anet_long_short/anet_gt_short_0.2_new.json'
res_gt_file = '/data/2/v-yale/ActionLocalization/data/anet_long_short/anet_gt_short_0.2_split4.json'
train_data_dir = '/data/2/v-yale/ActionLocalization/data/anet_long_short/split4/training'
val_data_dir = '/data/2/v-yale/ActionLocalization/data/anet_long_short/split4/validation'
flow_feat_dir = '/data/2/v-yale/MSM/v-yale/ActivityNet/cmcs_feat/ANET_FEATURES/anet_i3d_feature_25fps/flow-resize-step16'

train_set = os.listdir(train_data_dir)
train_set = [s[2:-7] for s in train_set]
val_set = os.listdir(val_data_dir)
val_set = [s[2:-7] for s in val_set]

gts = json.load(open(ori_gt_file, 'r'))
gt = gts['database']

print('original data number', len(gt.keys()))

gt_new = dict()
num_act = 0
for vid_name, vid_anns in gt.items():
    if vid_name in train_set:
        vid_anns['subset'] = 'training'
        anns = vid_anns['annotations']
        num_act += len(anns)
        # gt_new[vid_name] = vid_anns
    elif vid_name in val_set:
        vid_anns['subset'] = 'validation'
        anns = vid_anns['annotations']
        num_act += len(anns)
        # gt_new[vid_name] = vid_anns
    else:
        pass
        # # print('unsual', vid_name)
        # feat_file = os.path.join(flow_feat_dir, 'v_'+vid_name+'-flow.npz')
        # if os.path.exists(feat_file):
        #     print(vid_name, vid_anns['subset'])
        #     # print(feat_file)
        # assert vid_anns['subset'] == 'testing'
    gt_new[vid_name] = vid_anns
print('action instance number', num_act)
# print('result data number', len(gt_new.keys()))
# gts['database'] = gt_new
# outfile = open(res_gt_file, 'w')
# json.dump(gts, outfile)
# outfile.close()
