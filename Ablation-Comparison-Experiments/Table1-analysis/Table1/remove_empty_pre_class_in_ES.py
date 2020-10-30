from collections import defaultdict
import os


pre_ES_file = './pre_af/pre_ES.txt'


pre = []
with open(pre_ES_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        vid_name = line.strip().split()[0]
        start_t = float(line.strip().split()[1])
        end_t = float(line.strip().split()[2])
        cls_index = int(line.strip().split()[3])
        conf = float(line.strip().split()[4])
        pre.append([vid_name, start_t, end_t, cls_index, conf])


class_id = defaultdict(int)
class_file = os.path.join('./GT_test_dif_dur/ES/', 'detclasslist.txt')
with open(class_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # learn
        cid = int(line.strip().split()[0])
        cname = line.strip().split()[1]
        class_id[cid] = cname


out_dir = './af__remove_invalid_class'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

f_txt = os.path.join(out_dir , 'pre_ES.txt')
f = open(f_txt, 'w')

for i in range(len(pre)):
    if pre[i][3]  in class_id.keys():
        strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre[i][1]), float(pre[i][2]), pre[i][3], pre[i][4])
        f.write(strout)
f.close()
print()