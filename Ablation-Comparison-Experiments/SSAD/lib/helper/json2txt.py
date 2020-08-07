import json
import numpy as np
import matplotlib.pyplot as plt
import os


gt_json_file = '/data/2/v-yale/ActionLocalization/data/anet_minor/gt_mini.json'
subset = 'validation'
res_file = '/data/2/v-yale/ActionLocalization/data/anet_minor/gt_minor.txt'

gts = json.load(open(gt_json_file, 'r'))
gt = gts['database']


for vid_name, vid_anns in gt.items():
    if vid_anns['subset'] != subset:
        continue

    anns = vid_anns['annotations']
    for ann in anns:
        segment = ann['segment']
        label_tmp = ann['label']
        with open(res_file, 'a') as f:
            f.write('%s %.3f, %.3f, %s\n' % (vid_name, segment[0], segment[1], label_tmp))



# # prediction
# cls_name = 'Using uneven bars'
# save_dir = '/data/2/v-yale/materials/BSN_pred'
# gt_file = '/data/2/v-yale/ActionLocalization/lib/dataset/materials/activity_net.v1-3.min.json'
# pred_file = '/data/2/v-yale/ActionLocalization/lib/attempt/anet18_winner_validation.json'
#
#
# with open(gt_file, 'r') as f:
#     gts = json.load(f)
# gt = gts['database']
#
# preds = json.load(open(pred_file, 'r'))
# predictions = preds['results']
# pred_keys = predictions.keys()
#
#
# for vid_name, vid_anns in gt.items():
#     if vid_anns['subset'] != 'validation':
#         continue
#
#     anns = vid_anns['annotations']
#     label = anns[0]['label']
#     if label != cls_name:
#         continue
#
#     if vid_name not in pred_keys:
#         show_pred = False
#     else:
#         show_pred = True
#
#     for order, ann in enumerate(anns):
#         segment = ann['segment']
#         label_tmp = ann['label']
#         # print('%s, %.2f, %.2f, %s' % (vid_name, segment[0], segment[1], label_tmp))
#
#         if show_pred:
#             vid_preds = predictions[vid_name]
#             for i in range(5):
#                 vid_pred = vid_preds[i]
#                 segment = vid_pred['segment']
#                 label_tmp = vid_pred['label']
#                 score = vid_pred['score']
#                 print('%s, %.2f, %.2f, %s, %.2f' % (vid_name, segment[0], segment[1], label_tmp, score))
