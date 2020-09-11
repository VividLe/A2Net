import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from core.anchor_box_match import anchor_bboxes_encode, anchor_box_adjust
from core.loss import loss_function_ab, loss_function_af
from core.anchor_box_utils import result_process_ab, result_process_af
from core.prediction_box_match import reg2loc


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, train_loader, model, optimizer):
    model.train()
    loss_record = 0

    for feat_spa, feat_tem, boxes, action_num, cls_label_af, reg_label_af in train_loader:
        feature = torch.cat((feat_spa, feat_tem), dim=1)
        # feature = feat_spa
        optimizer.zero_grad()

        feature = feature.type_as(dtype)


        #####################################  anchor based
        boxes = boxes.type_as(dtype)
        actions_num = action_num.detach().numpy().astype(np.int)

        predictions = model(feature, is_ab=True)
        anchors_pred = torch.cat(predictions, dim=1)  # [batch, 505, 1]

        match_scores, match_masks, anchor_xs, anchor_ws, anchor_overlaps = \
            anchor_bboxes_encode(cfg, anchors_pred, boxes, actions_num)

        overlap_loss = loss_function_ab(anchor_overlaps, match_scores, match_masks, cfg)


        #####################################  anchor free
        cls_label_af = cls_label_af.type_as(dtypel)
        reg_label_af = reg_label_af.type_as(dtype)
        preds_cls, preds_reg = model(feature, is_ab=False)  # [batch, temporal_length, 2], [batch, temporal_length, 2]
        cls_loss_af, reg_loss_af = loss_function_af(cls_label_af, preds_cls, reg_label_af, preds_reg, cfg)

        loss_ab = overlap_loss * cfg.TRAIN.P_LOC_AB
        cls_loss_af = cls_loss_af * cfg.TRAIN.P_CLASS_AF
        reg_loss_af = reg_loss_af*cfg.TRAIN.P_LOC_AF

        loss_af = cls_loss_af  + reg_loss_af
        # loss_ab = 0
        loss = loss_af + loss_ab
        # print('cls_loss_af %f\t'%(cls_loss_af.item()), 'reg_loss_af %f\t'%(reg_loss_af.item()), 'overlap_loss %f\t'%(overlap_loss.item()))
        loss.backward()
        optimizer.step()
        loss_record = loss_record + loss.item()

    loss_avg = loss_record / len(train_loader)
    return loss_avg


def evaluation(val_loader, model, epoch, cfg):
    model.eval()

    out_df_ab = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS)
    out_df_af = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS)

    # we only use temporal feature, we do not use video length
    for feat_spa, feat_tem, start_frame, video_name in val_loader:
        start_frame = start_frame.detach().numpy()
        feature = torch.cat((feat_spa, feat_tem), dim=1)
        # feature = feat_spa
        feature = feature.type_as(dtype)


        ######################## anchor based
        predictions = model(feature, is_ab = True)
        anchors_pred = torch.cat(predictions, dim=1)  # [batch, 633, 1]

        anchors_overlap, anchors_xs, anchors_ws = anchor_box_adjust(cfg, anchors_pred)
        # print('anchors_overlap[1, -30]', anchors_overlap[1, -30:])
        anchors_overlap = anchors_overlap.detach().cpu().numpy()

        # convert box
        anchors_xmins = anchors_xs - anchors_ws / 2
        anchors_xmaxs = anchors_xs + anchors_ws / 2
        anchors_xmins = torch.squeeze(anchors_xmins)
        anchors_xmins = anchors_xmins.detach().cpu().numpy()
        anchors_xmaxs = torch.squeeze(anchors_xmaxs)
        anchors_xmaxs = anchors_xmaxs.detach().cpu().numpy()

        # convert results to df
        tmp_df_ab = result_process_ab(video_name, start_frame, anchors_overlap, anchors_xmins, anchors_xmaxs, cfg)
        out_df_ab = pd.concat([out_df_ab, tmp_df_ab])


        ############################## anchor free
        preds_cls, preds_reg = model(feature, is_ab=False)  # [batch, temporal_length, 2], [batch, temporal_length, 2]

        # fore-ground confidence
        m = nn.Softmax(dim=2).cuda()
        preds_cls = m(preds_cls)
        conf_score = preds_cls[:, :, 1]  # [batch, temporal_length]
        conf_score = conf_score.detach().cpu().numpy()

        # regression
        preds_loc = reg2loc(cfg, preds_reg)
        xmins = preds_loc[:, :, 0]
        xmins = xmins.detach().cpu().numpy()
        xmaxs = preds_loc[:, :, 1]
        xmaxs = xmaxs.detach().cpu().numpy()

        # convert results to df
        tmp_df_af = result_process_af(video_name, start_frame, conf_score, xmins, xmaxs, cfg)
        out_df_af = pd.concat([out_df_af, tmp_df_af])

    # save prediction csv file
    if cfg.SAVE_PREDICT_RESULT:
        predict_file_af = os.path.join(cfg.ROOT_DIR, cfg.TEST.PREDICT_CSV_FILE + '_af_' + str(epoch) + '.csv')
        print('predict_file', predict_file_af)

        tmp_af = out_df_af.sort_values(by='conf', ascending=False)

        tmp_af.to_csv(predict_file_af, index=False)

        predict_file_ab = os.path.join(cfg.ROOT_DIR, cfg.TEST.PREDICT_CSV_FILE + '_ab_' + str(epoch) + '.csv')
        print('predict_file', predict_file_ab)
        tmp_ab = out_df_ab.sort_values(by='conf', ascending=False)
        tmp_ab.to_csv(predict_file_ab, index=False)

    return out_df_af, out_df_ab

