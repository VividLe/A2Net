import argparse
import numpy as np
import os
from Evaluation.eval_detection import ANETdetection
import pickle


def parser_args():
    parser = argparse.ArgumentParser(description='Evaluate action detection performance')
    parser.add_argument('-gt_file',
                        default='/data/yangle/zt/ab_af_new_zt/lib/dataset/materials/activity_net.v1-3.min.json')
    # parser.add_argument('-prediction_file',
    #                     default='/data/2/v-yale/ActionLocalization/output/af_train/action_detection_20_gt_label_short.json')
    parser.add_argument('-prediction_dir', default='/data/yangle/zt/ab_af_new_zt/output/all512_seed_json/')
    parser.add_argument('-subset', default='validation')
    parser.add_argument('-postfix', default='_cuhk_label.json')
    parser.add_argument('-tiou_th', default=np.linspace(0.5, 0.95, 10))
    parser.add_argument('-verbose', default=True)
    parser.add_argument('-check_status', default=True)
    return parser.parse_args()


def eval_detections(prediction_dir, postfix, gt_file, subset, tiou_th, verbose, check_status):
    file_set = os.listdir(prediction_dir)
    file_set = [i for i in file_set if i.endswith(postfix)]
    file_set.sort()

    for file_name in file_set:
        print(file_name)
        prediction_file = os.path.join(prediction_dir, file_name)
        anet_detection = ANETdetection(gt_file, prediction_file, subset=subset,
                                       tiou_thresholds=tiou_th, verbose=verbose,
                                       check_status=check_status)
        anet_detection.evaluate()


def eval_detection(prediction_file, postfix, gt_file, subset, tiou_th, verbose, check_status):
    anet_detection = ANETdetection(gt_file, prediction_file, subset=subset,
                                   tiou_thresholds=tiou_th, verbose=verbose, check_status=check_status)

    anet_detection.evaluate()


if __name__ == '__main__':
    args = parser_args()
    eval_detections(args.prediction_dir, args.postfix, args.gt_file, args.subset, args.tiou_th, args.verbose, args.check_status)
