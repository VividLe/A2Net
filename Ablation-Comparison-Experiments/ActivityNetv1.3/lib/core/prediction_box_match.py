import math
import numpy as np
import torch

dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def locate_point(tem_length, stride):
    location = list()
    stride_l = math.floor(stride / 2)
    for i in range(tem_length):
        data = i * stride + stride_l
        location.append(data)
    return location


def location_point_back(cfg, to_tensor=False, to_duration=False):
    location = list()
    for layer_name in cfg.MODEL.LAYERS_NAME_AF:
        tem_length = cfg.MODEL.TEMPORAL_LENGTH[layer_name]
        stride = cfg.MODEL.TEMPORAL_STRIDE[layer_name]
        loc_layer = locate_point(tem_length, stride)
        location.extend(loc_layer)

    if to_duration:
        duration = list()
        start_time = 0
        end_time = 2
        for layer_name in cfg.MODEL.LAYERS_NAME_AF:
            tem_length = cfg.MODEL.TEMPORAL_LENGTH[layer_name]
            data = [[start_time, end_time] for i in range(tem_length)]
            duration.extend(data)
            # updata
            start_time = end_time
            end_time = end_time * 2
        print(len(duration))

    # convert to tensor
    if to_tensor:
        location_point = np.array(location)
        location_point = torch.from_numpy(location_point)
        location_point = torch.unsqueeze(location_point, dim=0)
        location_point = location_point.type_as(dtype)
        return location_point
    elif to_duration:
        return location, duration
    else:
        return location


# TODO: I can improve the efficiency here
def determine_location(cls_label, loc_label, location_point):
    # cls_label: (254); loc_label: [1, 2]; pred_reg: [254, 2]
    num_loc = cls_label.shape[0]
    assert num_loc == 254
    num_gt_locs = loc_label.shape[0]
    target_reg = torch.zeros((num_loc, 2)).type_as(dtype)

    for i in range(num_loc):
        cls_l = cls_label[i]
        # background
        if cls_l == 0:
            continue
        # foreground
        loc_tmp = location_point[i]

        # search gt locations to find out the desired one
        # TODO: dispose not found issue
        for j in range(num_gt_locs):
            act_start = loc_label[j, 0]
            act_end = loc_label[j, 1]
            if (loc_tmp >= act_start) and (loc_tmp <= act_end):
                # print('find action', act_start, loc_tmp, act_end)
                target_reg[i, 0] = loc_tmp - act_start
                target_reg[i, 1] = act_end - loc_tmp
                break

    return target_reg


def decode_location(cfg, cls_labels, loc_labels, actions_num):
    location_point = location_point_back(cfg, to_tensor=False)

    # dispose batch-by-batch
    target_regs = list()
    for i in range(actions_num.shape[0]):
        num_action = actions_num[i]
        cls_label = cls_labels[i, :]
        loc_label = loc_labels[i, :num_action, :]
        target_reg = determine_location(cls_label, loc_label, location_point)
        target_regs.append(torch.unsqueeze(target_reg, dim=0))
    target_regs = torch.cat(target_regs, dim=0)

    return target_regs


def reg2loc(cfg, pred_regs):
    # pred_regs [batch, 254, 2]
    num_batch = pred_regs.size()[0]
    location_point = location_point_back(cfg, to_tensor=True)  # [1, 254]
    location_point = location_point.repeat(num_batch, 1)
    pred_locs = torch.zeros(pred_regs.size()).type_as(dtype)

    # filter out
    num_pred = pred_regs.size(1)
    location_point = location_point[:, :num_pred].contiguous()
    # left boundary
    pred_locs[:, :, 0] = location_point - pred_regs[:, :, 0]
    # right boundary
    pred_locs[:, :, 1] = location_point + pred_regs[:, :, 1]

    return pred_locs


def reg2loc_loc(pred_regs, ctr_loc):
    pred_locs = torch.zeros(pred_regs.size()).type_as(dtype)
    pred_locs[:, :, 0] = ctr_loc - pred_regs[:, :, 0]
    pred_locs[:, :, 1] = ctr_loc + pred_regs[:, :, 1]
    return pred_locs


if __name__ == '__main__':
    from config import cfg, update_config
    cfg_file = '/data/2/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    location, duration = location_point_back(cfg, to_tensor=False, to_duration=True)
    i = 0
    for loc, dur in zip(location, duration):
        print(i, loc, dur)
        i += 1

