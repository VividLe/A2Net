import sys, os, errno
import numpy as np
import csv
import json
import copy
import argparse
from .Evaluation.eval_detection import ANETdetection


def nms(dets, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = dets[:, 2]
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def generate_classes(meta_file):
    data = json.load(open(meta_file))
    class_list = []
    for vid, vinfo in data['database'].items():
        for item in vinfo['annotations']:
            class_list.append(item['label'])

    class_list = list(set(class_list))
    class_list = sorted(class_list)
    classes = {0: 'Background'}
    for i, cls in enumerate(class_list):
        classes[i + 1] = cls
    return classes


def get_segments(data, thresh, framerate):
    segments = []
    vid = 'Background'
    find_next = False
    tmp = {'label': 0, 'score': 0, 'segment': [0, 0]}
    for l in data:
        # video name and sliding window length
        if "fg_name:" in l:
            vid = l.split('/')[-1]

        # frame index, time, confident score
        elif "frames:" in l:
            # dispose irregular string
            if '[[0 ' in l:
                l = l.replace('[[0 ', '[[  0 ')
            start_frame = int(l.split()[3])
            end_frame = int(l.split()[4])
            stride = int(l.split()[5].split(']')[0])

        elif "activity:" in l:
            label = int(l.split()[1])
            tmp['label'] = label
            find_next = True

        elif "im_detect" in l:
            return vid, segments

        elif find_next:
            try:
                left_frame = float(l.split()[0].split('[')[-1]) * stride + start_frame
                right_frame = float(l.split()[1]) * stride + start_frame
            except:
                left_frame = float(l.split()[1]) * stride + start_frame
                right_frame = float(l.split()[2]) * stride + start_frame

            try:
                score = float(l.split()[-1].split(']')[0])
            except:
                score = float(l.split()[-2])

            if (left_frame >= right_frame):
                print("???", l)
                continue

            if right_frame > end_frame:
                # print("right out", right_frame, end_frame)
                right_frame = end_frame

            left = left_frame / framerate
            right = right_frame / framerate
            if score > thresh:
                tmp1 = copy.deepcopy(tmp)
                tmp1['score'] = score
                tmp1['segment'] = [left, right]
                segments.append(tmp1)


def analysis_log(logfile, thresh, framerate):
    with open(logfile, 'r') as f:
        lines = f.read().splitlines()
    predict_data = []
    res = {}
    for l in lines:
        if "frames:" in l:
            predict_data = []
        predict_data.append(l)
        if "im_detect:" in l:
            vid, segments = get_segments(predict_data, thresh, framerate)
            if vid not in res:
                res[vid] = []
            res[vid] += segments
    return res


def select_top(segmentations, groundtruth_file, nms_thresh=0.99999, num_cls=0, topk=0):
    res = {}
    classes = generate_classes(groundtruth_file)
    for vid, vinfo in segmentations.items():
        # select most likely classes
        if num_cls > 0:
            ave_scores = np.zeros(201)
            for i in range(1, 201):
                ave_scores[i] = np.sum([d['score'] for d in vinfo if d['label'] == i])
            labels = list(ave_scores.argsort()[::-1][:num_cls])
        else:
            labels = list(set([d['label'] for d in vinfo]))

        # NMS
        res_nms = []
        for lab in labels:
            nms_in = [d['segment'] + [d['score']] for d in vinfo if d['label'] == lab]
            keep = nms(np.array(nms_in), nms_thresh)
            for i in keep:
                tmp = {'label': classes[lab], 'score': nms_in[i][2], 'segment': nms_in[i][0:2]}
                res_nms.append(tmp)

        # select topk
        scores = [d['score'] for d in res_nms]
        sortid = np.argsort(scores)[-topk:]
        res[vid] = [res_nms[id] for id in sortid]
    return res


class analyzor():

    def __init__(self, framerate=3, subset='validation'):
        self.prediction_name = 'results.json'
        self.framerate = framerate
        # filter those dets low than the thresh
        self.thresh = 0.005
        self.nms_thresh = 0.3
        self.topk = 200
        # select most likely classes, the meaning of this param?
        self.num_cls = 0
        self.THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.prediction_file = self.THIS_DIR + self.prediction_name
        self.ground_truth_file = self.THIS_DIR + '/Evaluation/data/activity_net.v1-3.min.json'
        # self.ground_truth_file = self.THIS_DIR + '/activity_net-mini.v1-3.json'
        self.subset = subset
        self.tiou_thresholds = np.linspace(0.5, 0.95, 10)
        self.verbose = True
        self.check_status = True

    def evaluate(self):
        anet_detection = ANETdetection(self.ground_truth_file, self.prediction_file,
                                       subset=self.subset, tiou_thresholds=self.tiou_thresholds,
                                       verbose=self.verbose, check_status=self.check_status)
        status = anet_detection.evaluate()
        return status

    def analyze(self, log_file):
        segmentations = analysis_log(log_file, thresh=self.thresh, framerate=self.framerate)
        segmentations = select_top(segmentations, self.ground_truth_file, nms_thresh=self.nms_thresh, num_cls=self.num_cls, topk=self.topk)

        res = {'version': 'VERSION 1.3',
               'external_data': {'used': True, 'details': 'C3D pre-trained on sport-1M training set'},
               'results': {}}
        for vid, vinfo in segmentations.items():
            res['results'][vid] = vinfo

        with open(self.prediction_file, 'w') as outfile:
            json.dump(res, outfile)
        score = self.evaluate()

        return score


# if __name__ == '__main__':
#     log_file = '/data/home/v-yale/ActionBaseline_test_acti6/output/c3d/activitynet/model_11_09444.txt'
#     ana = analyzor(framerate=6)
#     score = ana.analyze(log_file)


