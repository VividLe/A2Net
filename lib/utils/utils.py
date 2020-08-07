import torch
import os
import numpy as np
import random
import json
import shutil

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def save_model(cfg, epoch, model, optimizer):
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.MODEL_DIR, 'model_'+str(epoch).zfill(2)+'.pth')
    save_checkpoint(state, save_file)
    print('save model: %s' % save_file)
    return


def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def expand_vid_len(video_len, tmp_data):
    video_len_exp = np.expand_dims(video_len, axis=1)
    video_len_exp = np.tile(video_len_exp, (1, tmp_data.shape[1]))
    return video_len_exp


def prepare_output_file(result_dict, result_file):
    # prepare for output
    output_dict = {'version': 'VERSION 1.3', 'results': result_dict, 'external_data': {}}
    outfile = open(result_file, 'w')
    json.dump(output_dict, outfile)
    outfile.close()


def backup_codes(root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    print('codes backup')
