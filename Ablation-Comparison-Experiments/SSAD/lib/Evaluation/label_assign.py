import json

gt_file = '/data/home/v-yale/ActionLocalization/lib/dataset/materials/activity_net.v1-3.min.json'
pred_file = '/data/home/v-yale/ActionLocalization/output/action_detection_60.json'
output_file = '/data/home/v-yale/ActionLocalization/output/action_detection_60_gt_label.json'

with open(gt_file, 'r') as f:
    gts = json.load(f)
gt = gts['database']


with open(pred_file, 'r') as f:
    results = json.load(f)
predictions = results['results']


for vid_name, preds in predictions.items():
    anns = gt[vid_name]['annotations']
    gt_label = anns[0]['label']
    print(vid_name, gt_label)
    for pred in preds:
        pred['label'] = gt_label

output_dict = {'version': 'VERSION 1.3', 'results': predictions, 'external_data': {}}

outfile = open(output_file, 'w')
json.dump(output_dict, outfile)
outfile.close()

