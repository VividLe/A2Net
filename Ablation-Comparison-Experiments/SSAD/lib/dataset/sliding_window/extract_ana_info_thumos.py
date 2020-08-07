import os
from collections import defaultdict
import pandas as pd
import pickle


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
        with open('segment_annotation.txt', 'w') as f:
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


if __name__ == '__main__':
    # split in {'val', 'test'}
    split = 'val'
    ann_dir = '/data/home/v-yale/ActionLocalization/lib/dataset/materials/thumos_ann/annotation_val'
    vid_data_dir = '/data/home/v-yale/MSMdata/v-yale/BasicDataset/THUMOS/THUMOS14/I3D_feature/val_data'

    class_list = [0, 7, 9, 12, 21, 22, 23, 24, 26, 31, 33, 36, 40, 45, 51, 68, 79, 85, 92, 93, 97]
    type_list = ['Ambiguous', 'BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

    vid_names_25fps = ['video_validation_0000311', 'video_validation_0000420', 'video_validation_0000666', 'video_validation_0000419', 'video_validation_0000484', 'video_validation_0000413']
    dict_cls_type = dict(zip(class_list, type_list))
    # directory to .mp4 files

    extractor = Extractor(split, ann_dir, use_ambiguous=True)
    gt_segment = extractor.annotation_parser()
    # pickle.dump(gt_segment, open('ann_val.pkl', 'wb'))

    video_names = list()
    type_v = list()
    type_idx = list()
    start_frame = list()
    end_frame = list()
    frame_num = list()

    for vid_name, vid_anns in gt_segment.items():
        print(vid_name)
        if vid_name in vid_names_25fps:
            fps = 25
        else:
            fps = 30
        for ann in vid_anns:
            video_names.append(vid_name)
            type_v.append(dict_cls_type[ann[2]])
            type_idx.append(ann[2])
            start_frame.append(int(ann[0] * fps))
            end_frame.append(int(ann[1] * fps))

            frames_path = os.path.join(vid_data_dir, vid_name)
            frames_set = os.listdir(frames_path)
            assert len(frames_set) % 3 == 0
            frame_num.append(int(len(frames_set) / 3))

    dic_inf = dict()
    dic_inf['video'] = video_names
    dic_inf['type'] = type_v
    dic_inf['type_idx'] = type_idx
    dic_inf['startFrame'] = start_frame
    dic_inf['endFrame'] = end_frame
    dic_inf['frame_num'] = frame_num

    df = pd.DataFrame(dic_inf, columns=['video', 'type', 'type_idx', 'startFrame', 'endFrame', 'frame_num'])
    df.sort_values(['video', 'type_idx', 'startFrame'], inplace=True)
    df.to_csv('thumos14_test_annotation.csv', encoding='utf-8', index=False)

