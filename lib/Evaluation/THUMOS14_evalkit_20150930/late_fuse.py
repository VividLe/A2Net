import pandas as pd
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-spatial_path', default='/disk/peiliang/ActLocSSAD_complete/output_all/11_08/spa/prediction_28.csv')
    parser.add_argument('-temporal_path', default='/disk/peiliang/ActLocSSAD_complete/output_all/11_08/tem/prediction_30.csv')
    parser.add_argument('-fuse_path', default='/disk/peiliang/ActLocSSAD_complete/output_fuse/spa_tem/later_fuse.csv')
    args = parser.parse_args()

    return args
def fuse_two_stream(spatial_path, temporal_path):
    temporal_df = pd.read_csv(temporal_path)
    spatial_df = pd.read_csv(spatial_path)
    out_df = temporal_df
    out_df['conf'] = temporal_df.conf.values[:] * 2 / 3 + spatial_df.conf.values * 1 / 3
    out_df['xmin'] = temporal_df.xmin.values[:] * 2 / 3 + spatial_df.xmin.values * 1 / 3
    out_df['xmax'] = temporal_df.xmax.values[:] * 2 / 3 + spatial_df.xmax.values * 1 / 3
    out_df['score_0'] = temporal_df.score_0.values[:] * 2 / 3 + spatial_df.score_0.values * 1 / 3
    out_df['score_1'] = temporal_df.score_1.values[:] * 2 / 3 + spatial_df.score_1.values * 1 / 3
    out_df['score_2'] = temporal_df.score_2.values[:] * 2 / 3 + spatial_df.score_2.values * 1 / 3
    out_df['score_3'] = temporal_df.score_3.values[:] * 2 / 3 + spatial_df.score_3.values * 1 / 3
    out_df['score_4'] = temporal_df.score_4.values[:] * 2 / 3 + spatial_df.score_4.values * 1 / 3
    out_df['score_5'] = temporal_df.score_5.values[:] * 2 / 3 + spatial_df.score_5.values * 1 / 3
    out_df['score_6'] = temporal_df.score_6.values[:] * 2 / 3 + spatial_df.score_6.values * 1 / 3
    out_df['score_7'] = temporal_df.score_7.values[:] * 2 / 3 + spatial_df.score_7.values * 1 / 3
    out_df['score_8'] = temporal_df.score_8.values[:] * 2 / 3 + spatial_df.score_8.values * 1 / 3
    out_df['score_9'] = temporal_df.score_9.values[:] * 2 / 3 + spatial_df.score_9.values * 1 / 3
    out_df['score_10'] = temporal_df.score_10.values[:] * 2 / 3 + spatial_df.score_10.values * 1 / 3
    out_df['score_11'] = temporal_df.score_11.values[:] * 2 / 3 + spatial_df.score_11.values * 1 / 3
    out_df['score_12'] = temporal_df.score_12.values[:] * 2 / 3 + spatial_df.score_12.values * 1 / 3
    out_df['score_13'] = temporal_df.score_13.values[:] * 2 / 3 + spatial_df.score_13.values * 1 / 3
    out_df['score_14'] = temporal_df.score_14.values[:] * 2 / 3 + spatial_df.score_14.values * 1 / 3
    out_df['score_15'] = temporal_df.score_15.values[:] * 2 / 3 + spatial_df.score_15.values * 1 / 3
    out_df['score_16'] = temporal_df.score_16.values[:] * 2 / 3 + spatial_df.score_16.values * 1 / 3
    out_df['score_17'] = temporal_df.score_17.values[:] * 2 / 3 + spatial_df.score_17.values * 1 / 3
    out_df['score_18'] = temporal_df.score_18.values[:] * 2 / 3 + spatial_df.score_18.values * 1 / 3
    out_df['score_19'] = temporal_df.score_19.values[:] * 2 / 3 + spatial_df.score_19.values * 1 / 3
    out_df['score_20'] = temporal_df.score_20.values[:] * 2 / 3 + spatial_df.score_20.values * 1 / 3

    out_df = out_df[out_df.score_0 < 0.99]
    out_df.to_csv(args.fuse_path, index=False)
    return out_df

if __name__ == '__main__':
    args = args_parser()
    fuse_two_stream(args.spatial_path, args.temporal_path)