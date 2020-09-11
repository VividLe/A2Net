import torch
import numpy as np

from core.anchor_box_utils import jaccard_with_anchors


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def default_boxes_layer(layer_steps, scale, a_ratios):
    '''
    Given anchor num, scale and aspect_ratios
    generate default anchor boxes
    '''
    width_set = [scale * ratio for ratio in a_ratios]
    center_set = [0.5/layer_steps + 1./layer_steps*i for i in range(layer_steps)]
    width_default = list()
    center_default = list()

    for i in range(layer_steps):
        for j in range(len(a_ratios)):
            width_default.append(width_set[j])
            center_default.append(center_set[i])
    width_default = np.array(width_default)
    width_default = torch.from_numpy(width_default).type_as(dtype)
    center_default = np.array(center_default)
    center_default = torch.from_numpy(center_default).type_as(dtype)

    return width_default, center_default


def default_boxes(cfg):
    '''
    generate default boxes for multiple layers
    '''
    dboxes_w_list = list()
    dboxes_x_list = list()
    for layer_name in cfg.MODEL.LAYERS_NAME_AB:
        depth = cfg.MODEL.NUM_ANCHORS[layer_name]
        dboxes_w, dboxes_x = default_boxes_layer(depth, 1.0 / depth, cfg.MODEL.ASPECT_RATIOS[layer_name])
        dboxes_w_list.append(torch.unsqueeze(dboxes_w, dim=0))
        dboxes_x_list.append(torch.unsqueeze(dboxes_x, dim=0))
    dboxes_ws = torch.cat(dboxes_w_list, dim=1)
    dboxes_xs = torch.cat(dboxes_x_list, dim=1)  # [1, 633]
    return dboxes_ws, dboxes_xs


def anchor_box_adjust(cfg, anchors):
    '''
    Reture overlap prediction and default box
    '''
    dboxes_ws, dboxes_xs = default_boxes(cfg)
    anchors_overlap = anchors[:, :, 0]

    return anchors_overlap, dboxes_xs, dboxes_ws


def match_anchor_box(cfg, num_actions, anchor_xs, anchor_ws, gboxes):
    '''
    match anchor with ground truth
    Type1. match each gt box to the default box with the best jaccard overlap
    Type2. match default box to any ground truth with jaccard overlap higher than a threshold (0.5)
    '''

    num_anchors = anchor_xs.size()
    match_scores = torch.zeros(num_anchors).type_as(dtype)
    match_masks = torch.zeros(num_anchors).type_as(dtype)

    for idx in range(num_actions):
        box_min = gboxes[idx, 0]
        box_max = gboxes[idx, 1]

        anchors_min = anchor_xs - anchor_ws / 2
        anchors_max = anchor_xs + anchor_ws / 2
        jaccards = jaccard_with_anchors(anchors_min, anchors_max, box_min, box_max)

        mask_greater_j = torch.ge(jaccards, match_scores)
        # Type1 match
        if jaccards.max() < cfg.TRAIN.MATCH_TH:
            mask_match = mask_greater_j & torch.eq(jaccards, jaccards.max())
        # Type2 match
        else:
            mask_match = mask_greater_j & torch.gt(jaccards, cfg.TRAIN.MATCH_TH)

        # update values using mask
        mask_greater_j_f = mask_greater_j.type_as(dtype)
        match_scores = mask_greater_j_f * jaccards + (1 - mask_greater_j_f) * match_scores
        mask_match_f = mask_match.type_as(dtype)
        match_masks = mask_match_f + (1 - mask_match_f) * match_masks

    return match_scores, match_masks


def anchor_bboxes_encode(cfg, anchors, gboxes, g_action_nums):
    '''
    Produce matched ground truth with each adjusted anchors
    '''

    anchors_overlap, anchors_xs, anchors_ws = anchor_box_adjust(cfg, anchors)
    anchor_xs = torch.squeeze(anchors_xs)
    anchor_ws = torch.squeeze(anchors_ws)

    # calculate batch-by-batch
    batch_match_scores = list()
    batch_match_masks = list()

    for i in range(g_action_nums.shape[0]):
        num_action = g_action_nums[i]
        gbox = gboxes[i, :num_action, :]

        # match overlap score here
        match_scores, match_masks = match_anchor_box(cfg, num_action, anchor_xs, anchor_ws, gbox)
        batch_match_scores.append(torch.unsqueeze(match_scores, dim=0))
        batch_match_masks.append(torch.unsqueeze(match_masks, dim=0))

    batch_match_scores = torch.cat(batch_match_scores, dim=0)
    batch_match_masks = torch.cat(batch_match_masks, dim=0)

    return batch_match_scores, batch_match_masks, anchors_xs, anchors_ws, anchors_overlap


if __name__ == '__main__':
    from config import cfg, update_config
    cfg_file = '/data/home/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    default_boxes(cfg)
