import os
from collections import defaultdict
import argparse
import shutil

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

    print()

    class_id = defaultdict(int)
    class_file = os.path.join('./thumos_ann/test', 'detclasslist.txt')
    with open(class_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # learn
            cid = int(line.strip().split()[0])
            cname = line.strip().split()[1]
            class_id[cid] = cname


    d_list = [0, d1, d2, d3, d4, d5]
    dir_name_list = ['ES', 'S', 'M', 'L', 'EL']
    for d in range(len(d_list)-1):
        for i in range(d_list[d],d_list[d+1]):   ##################################################################################
            class_name = class_id[action[i][3]]

            out_dir = os.path.join('GT_test_dif_dur', dir_name_list[d])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.copy('./thumos_ann/test/Ambiguous_test.txt', out_dir)
            shutil.copy('./thumos_ann/test/detclasslist.txt', out_dir)
            out_file = os.path.join(out_dir, class_name + '_test.txt')
            with open(out_file,'a') as f:
                stout = '%s\t%.3f\t%.3f\n'%(action[i][0], action[i][1], action[i][2])
                f.write(stout)



if __name__ == '__main__':
    args = args_parser()
    extract(args)

