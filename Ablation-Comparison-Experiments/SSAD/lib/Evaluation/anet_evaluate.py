import argparse
import numpy as np
from Evaluation.eval_detection import ANETdetection


parser = argparse.ArgumentParser()
parser.add_argument('-gt_file', default='/data/home/v-yale/ActionLocalization/output/gt_mini.json')
parser.add_argument('-prediction_file', default='/data/home/v-yale/ActionLocalization/output/action_detection_500.json')
# parser.add_argument('-gt_file', default='./Evaluation/data/activity_net.v1-3.min.json')
# parser.add_argument('-prediction_file', default='/data/home/v-yale/ActionLocalization/output/action_detection_ratio_120.json')
parser.add_argument('-subset', default='training')
parser.add_argument('-tiou_th', default=np.linspace(0.5, 0.95, 10))
parser.add_argument('-verbose', default=True)
parser.add_argument('-check_status', default=True)
args = parser.parse_args()


anet_detection = ANETdetection(args.gt_file, args.prediction_file, subset=args.subset,
                               tiou_thresholds=args.tiou_th, verbose=args.verbose, check_status=args.check_status)
anet_detection.evaluate()

