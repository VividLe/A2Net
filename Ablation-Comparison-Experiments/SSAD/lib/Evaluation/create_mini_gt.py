import json

video_name_set = ['0MyaFtXcTKI', '0e7-mNDgIXw']
gt_file = '/data/home/v-yale/ActionLocalization/lib/dataset/materials/activity_net.v1-3.min.json'
output_file = '/data/home/v-yale/ActionLocalization/output/gt_mini.json'

with open(gt_file, 'r') as f:
    gts = json.load(f)
gt = gts['database']

res = dict()
for vid_name in video_name_set:
    anns = gt[vid_name]
    res[vid_name] = anns
gts['database'] = res

outfile = open(output_file, 'w')
json.dump(gts, outfile)
outfile.close()

