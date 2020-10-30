import os
from collections import defaultdict
import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser(description='Extract thumos ann')
    parser.add_argument('-split', default='test')
    parser.add_argument('-ann_dir', default='./thumos_ann/test')
    parser.add_argument('-class_list', default=[0, 7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97])
    parser.add_argument('-type_list', default=['Ambiguous', 'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                        'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                        'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                        'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'])
    parser.add_argument('-vid_names_25fps', default=['video_test_0000950', 'video_test_0001058', 'video_test_0001195', 'video_test_0001255', 'video_test_0001459'])
    parser.add_argument('-vid_names_24fps', default=['video_test_0001207'])

    parser.add_argument('-pre_txt', default = './action_detection_af38.txt')
    parser.add_argument('-out_dir', default = './pre_af')
    args = parser.parse_args()
    return args


class Extractor():
    def __init__(self, split, ann_dir, use_ambiguous=False, num_class=20):
        self.split = split
        self.use_ambiguous = use_ambiguous
        self.num_class = num_class
        self.ann_dir = ann_dir

    def id_and_name(self):
        '''
        parse detclasslist.txt, obtain class name and index
        '''
        # learn
        class_id = defaultdict(int)
        class_file = os.path.join(self.ann_dir, 'detclasslist.txt')
        with open(class_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # learn
                cid = int(line.strip().split()[0])
                cname = line.strip().split()[1]
                class_id[cname] = cid
        if self.use_ambiguous:
            class_id['Ambiguous'] = self.num_class + 1
        return class_id

    def record_annotation(self, segment):
        '''
        record annotation in txt file
        '''
        keys = list(segment.keys())
        keys.sort()
        with open('segment_annotation_val.txt', 'w') as f:
            for k in keys:
                f.write('{}\n{}\n\n'.format(k, segment[k]))

    def annotation_parser(self, show=True):
        class_id = self.id_and_name()
        segment = dict()

        for cname in class_id.keys():
            file = '{}_{}.txt'.format(cname, self.split)
            with open(os.path.join(self.ann_dir, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    vid_name = line.strip().split()[0]
                    start_t = float(line.strip().split()[1])
                    end_t = float(line.strip().split()[2])

                    if vid_name in segment.keys():
                        segment[vid_name].append([start_t, end_t, class_id[cname]])
                    else:
                        # create initial list
                        segment[vid_name] = [[start_t, end_t, class_id[cname]]]
        # sort the segment according to start time
        for vid in segment.keys():
            segment[vid].sort(key=lambda x: x[0])

        # record segment annotation
        if show:
            self.record_annotation(segment)

        return segment

def take(elem):
    return elem[4]

def extract(args):
    extractor = Extractor(args.split, args.ann_dir, use_ambiguous=False)
    gt_segment = extractor.annotation_parser(show=False)

    # segments = []
    action = []
    for vid_name, vid_anns in gt_segment.items():
        for ann in vid_anns:
            # segments.append([vid_name, ann[1]-ann[0], ann[2]])
            action.append([vid_name, ann[0], ann[1], ann[2], ann[1]-ann[0]])
    action.sort(key=take)
    # action.sort(key=takeSencond)
    num = len(action)
    d1  =int(num*0.2)
    d2  =int(num*0.4)
    d3  =int(num*0.6)
    d4  =int(num*0.8)
    d5  =num


    print('0.2:\t',action[d1][4],int(num*0.2))
    print('0.4:\t',action[d2][4],int(num*0.4))
    print('0.6:\t',action[d3][4],int(num*0.6))
    print('0.8:\t',action[d4][4],int(num*0.8))
    print('1.0:\t',action[-1][4], num)


    pre = []
    pre_txt = args.pre_txt
    with open(pre_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid_name = line.strip().split()[0]
            start_t = float(line.strip().split()[1])
            end_t = float(line.strip().split()[2])
            cls_index = int(line.strip().split()[3])
            conf = float(line.strip().split()[4])
            pre.append([vid_name, start_t, end_t, cls_index, conf])

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    f_None_txt = os.path.join(out_dir, 'pre_None.txt')
    f_None = open(f_None_txt, 'w')

    f_ES_txt = os.path.join(out_dir, 'pre_ES.txt')
    f_ES = open(f_ES_txt, 'w')

    f_S_txt = os.path.join(out_dir, 'pre_S.txt')
    f_S = open(f_S_txt, 'w')

    f_M_txt = os.path.join(out_dir, 'pre_M.txt')
    f_M = open(f_M_txt, 'w')

    f_L_txt = os.path.join(out_dir, 'pre_L.txt')
    f_L = open(f_L_txt, 'w')

    f_EL_txt = os.path.join(out_dir, 'pre_EL.txt')
    f_EL = open(f_EL_txt, 'w')

    for i in range(len(pre)):
        print(len(pre),':',i)
        iou_max = 0
        iou_max_index = None
        iou = 0
        pre_l = pre[i][1]
        pre_r = pre[i][2]
        for j in range(len(action)):
            if pre[i][0] != action[j][0]:
                continue

            GT_l = action[j][1]
            GT_r = action[j][2]

            inter_xmin = np.maximum(pre_l, GT_l)
            inter_xmax = np.minimum(pre_r, GT_r)
            inter_len = np.maximum(inter_xmax - inter_xmin, 0.)
            union_len = np.maximum(pre_r, GT_r) - np.minimum(pre_l, GT_l)
            iou = np.divide(inter_len, union_len)
            if iou > iou_max:
                iou_max = iou
                iou_max_index = j
        if iou_max_index == None:
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_None.write(strout)
        elif (iou_max_index >= 0 ) and (iou_max_index < d1):
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_ES.write(strout)
        elif (iou_max_index >= d1 ) and (iou_max_index < d2):
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_S.write(strout)
        elif (iou_max_index >= d2 ) and (iou_max_index < d3):
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_M.write(strout)
        elif (iou_max_index >= d3 ) and (iou_max_index < d4):
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_L.write(strout)
        elif (iou_max_index >= d4 ) and (iou_max_index <= d5):
            strout = '%s\t%.3f\t%.3f\t%d\t%.4f\n' % (pre[i][0], float(pre_l), float(pre_r), pre[i][3], pre[i][4])
            f_EL.write(strout)
    f_None.close()
    f_ES.close()
    f_S.close()
    f_M.close()
    f_L.close()
    f_EL.close()

    print()





if __name__ == '__main__':
    args = args_parser()
    extract(args)

