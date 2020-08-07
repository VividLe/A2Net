import json

gt_file = '/home/yangle/yangle/zt/SSAD_anet_zt/lib/dataset/materials/activity_net.v1-3.min.json'
with open(gt_file, 'r') as f:
    gts = json.load(f)
gt = gts['database']
for vid_name, vid_anns in gt.items():
    anns = vid_anns['annotations']
    for i in range(len(anns)):
        if i == 0:
            ann_label_check = anns[i]['label']
        else:
            if ann_label_check != anns[i]['label']:
                print(vid_name, ann_label_check, 'vs',anns[i]['label'])

print()