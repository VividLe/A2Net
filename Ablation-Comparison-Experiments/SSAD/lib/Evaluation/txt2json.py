import argparse
import json
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-txt_file', default='action_detection_100.txt')
parser.add_argument('-idx_name_file', default='./idx_name_dict.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    idx_name_dict = pickle.load(open(args.idx_name_file, 'rb'))
    # for k, v in idx_name_dict.items():
    #     print(k, v,)
    with open(args.txt_file, 'r') as f:
        lines = f.readlines()

    predictions = dict()

    for line in lines:
        datas = line.split('\t')
        vid_name, start_t, end_t, vid_idx, conf_score = datas

        if not vid_name in predictions.keys():
            preds = list()
            pred = dict()
            pred['label'] = idx_name_dict[int(vid_idx)+1]
            pred['segment'] = [float(start_t), float(end_t)]
            pred['score'] = float(conf_score[:-1])
            preds.append(pred)
            predictions[vid_name] = preds
        else:
            preds = predictions[vid_name]
            pred = dict()
            pred['label'] = idx_name_dict[int(vid_idx)+1]
            pred['segment'] = [float(start_t), float(end_t)]
            pred['score'] = float(conf_score[:-1])
            preds.append(pred)
            predictions[vid_name] = preds
    output_dict = {'version': 'VERSION 1.3', 'results': predictions, 'external_data': {}}

    outfile = open('SSAD_anet.json', 'w')
    json.dump(output_dict, outfile)
    outfile.close()

