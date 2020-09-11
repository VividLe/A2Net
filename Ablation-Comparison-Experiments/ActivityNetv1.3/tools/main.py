import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import _init_paths
from config import cfg
from utils.utils import fix_random_seed
from config import update_config
import pprint
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import shutil
import pandas as pd

from dataset.TALDataset import TALDataset
from models.ssad import LocNet
from core.function import train, evaluation
from core.post_process import post_process
from core.anchor_box_utils import weight_init
from utils.utils import save_model, decay_lr


def parse_args():
    parser = argparse.ArgumentParser(description='SSAD temporal action localization')
    parser.add_argument('--cfg', type=str, help='experiment config file', default='../experiments/ssad.yaml')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)
    fix_random_seed(cfg.SEED)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLE

    # prepare output directory
    output_dir = os.path.join(cfg.ROOT_DIR, cfg.TRAIN.MODEL_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # copy config ssad.yaml file to output directory
    cfg_file = os.path.join(output_dir, args.cfg.split('/')[-1])
    shutil.copyfile(args.cfg, cfg_file)

    # data loader
    # Notice: we discard the last data
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True, drop_last=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
    val_dset = TALDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)

    model = LocNet(cfg)
    model.apply(weight_init)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH+1):
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        loss_train = train(cfg, train_loader, model, optimizer)
        print('epoch %d: loss: %f' % (epoch, loss_train))
        with open(os.path.join(cfg.ROOT_DIR, cfg.TRAIN.LOG_FILE), 'a') as f:
            f.write("epoch %d, loss: %.4f\n" % (epoch, loss_train))

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            # model
            weight_file = save_model(cfg, epoch=epoch, model=model, optimizer=optimizer)
            out_df_af, out_df_ab = evaluation(val_loader, model, epoch, cfg)
            out_df_ab['conf'] = out_df_ab['conf'] * cfg.TEST.CONCAT_AB
            out_df = pd.concat([out_df_af, out_df_ab])
            post_process(out_df, epoch, cfg, is_soft_nms=False)


if __name__ == '__main__':
    main()

