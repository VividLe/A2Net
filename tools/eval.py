import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import _init_paths
from config import cfg
from utils.utils import fix_random_seed
from config import update_config
import pprint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import time

from dataset.TALDataset import TALDataset
from models.a2net import LocNet
from core.function import train, evaluation
from core.post_process import final_result_process
from core.anchor_box_utils import weight_init
from utils.utils import save_model, decay_lr, backup_codes


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='../experiments/anet/A2Net.yaml')
    parser.add_argument('--weight_file', type=str, help='experiment config file', default='../checkpoints/ActivityNet1.3.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)
    # create output directory
    if cfg.BASIC.CREATE_OUTPUT_DIR:
        out_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    # copy config file
    if cfg.BASIC.BACKUP_CODES:
        backup_dir = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'code')
        backup_codes(cfg.BASIC.ROOT_DIR, backup_dir, cfg.BASIC.BACKUP_LISTS)
    fix_random_seed(cfg.BASIC.SEED)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # data loader
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.DATASET.PIN_MEMORY)

    model = LocNet(cfg)
    model.apply(weight_init)
    model.cuda()

    #evaluate existing model
    weight_file = '/disk3/zt/code/2_TIP_rebuttal/2_A2Net/output/thumos/output_toy/model_100.pth'
    epoch = 4
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model'])
    out_df_ab, out_df_af = evaluation(val_loader, model, epoch, cfg)

    '''
    flag:
    0: jointly consider out_df_ab and out_df_af
    1: only consider out_df_ab
    2: only consider out_df_af    
    '''
    # evaluate both branch
    out_df_list = [out_df_ab, out_df_af]
    final_result_process(out_df_list, epoch, cfg, flag=0)
    # # only evaluate anchor-based branch
    # final_result_process(out_df_ab, epoch, cfg, flag=1)
    # # only evaluate anchor-free branch
    # final_result_process(out_df_af, epoch, cfg, flag=2)


if __name__ == '__main__':
    main()


