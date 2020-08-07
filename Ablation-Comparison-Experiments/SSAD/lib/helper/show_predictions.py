import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


# cls_name_set = ['Drinking coffee', 'Shot put', 'Throwing darts', 'Smoking a cigarette']
save_dir = '/data/2/v-yale/ActionLocalization/output/anet_10s/show_pred'
gt_file = '/data/2/v-yale/ActionLocalization/data/anet_10s/mini_gt.json'
pred_file = '/data/2/v-yale/ActionLocalization/output/anet_10s/action_detection_4_cuhk_label_100_top.json'
name_idx_file = '/data/2/v-yale/ActionLocalization/lib/dataset/materials/name_idx_dict.pkl'
# pred_file = '/data/2/v-yale/ActionLocalization/lib/attempt/anet18_winner_validation.json'


def segment2idx(start_time, end_time, duration, value):
    start_idx = round(start_time / duration * 100)
    end_idx = round(end_time / duration * 100)

    gt = np.zeros(100)
    gt[start_idx:end_idx] = value
    return gt


with open(gt_file, 'r') as f:
    gts = json.load(f)
gt_ann = gts['database']

name_idx_dict = pickle.load(open(name_idx_file, 'rb'))
cls_name_set = list(name_idx_dict.keys())

preds = json.load(open(pred_file, 'r'))
predictions = preds['results']
pred_keys = predictions.keys()

coord = np.array(list(range(1, 101)))
gt_value = 0.5
pred1_value = 0.6
pred2_value = 0.7
pred3_value = 0.8
pred4_value = 0.9
pred5_value = 1.0


for cls_name in cls_name_set:
    datas = cls_name.split(' ')
    cls_name_seq = datas[0]
    for d in datas[1:]:
        cls_name_seq = cls_name_seq + '_' + d

    cls_save_dir = os.path.join(save_dir, cls_name_seq)
    if not os.path.exists(cls_save_dir):
        os.makedirs(cls_save_dir)

    for vid_name, vid_anns in gt_ann.items():
        if vid_anns['subset'] != 'validation':
            continue

        anns = vid_anns['annotations']
        label = anns[0]['label']
        if label != cls_name:
            continue

        if vid_name not in pred_keys:
            show_pred = False
        else:
            show_pred = True
            # continue
        print(vid_name)

        duration = vid_anns['duration']
        for order, ann in enumerate(anns):
            segment = ann['segment']
            gt = segment2idx(segment[0], segment[1], duration, gt_value)
            plt.plot(coord, gt, label='gt')

            if show_pred:
                vid_preds = predictions[vid_name]
                pred1 = segment2idx(vid_preds[0]['segment'][0], vid_preds[0]['segment'][1], duration, pred1_value)
                pred2 = segment2idx(vid_preds[1]['segment'][0], vid_preds[1]['segment'][1], duration, pred2_value)
                pred3 = segment2idx(vid_preds[2]['segment'][0], vid_preds[2]['segment'][1], duration, pred3_value)
                pred4 = segment2idx(vid_preds[3]['segment'][0], vid_preds[3]['segment'][1], duration, pred4_value)
                pred5 = segment2idx(vid_preds[4]['segment'][0], vid_preds[4]['segment'][1], duration, pred5_value)

                plt.plot(coord, pred1, label='pred1')
                plt.plot(coord, pred2, label='pred2')
                plt.plot(coord, pred3, label='pred3')
                plt.plot(coord, pred4, label='pred4')
                plt.plot(coord, pred5, label='pred5')

            plt.legend()

            save_name = 'v_' + vid_name + '_' + str(order).zfill(2) + '.jpg'
            plt.savefig(os.path.join(cls_save_dir, save_name))
            plt.clf()
