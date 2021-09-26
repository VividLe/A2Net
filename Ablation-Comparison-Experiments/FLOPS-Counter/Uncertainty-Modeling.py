'''
Weakly-supervised Temporal Action Localization by Uncertainty Modeling
'''

import torch
import torch.nn as nn
import numpy as np


class UM_loss(nn.Module):
    def __init__(self, alpha, beta, margin):
        super(UM_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_criterion = nn.BCELoss()

    def forward(self, score_act, score_bkg, feat_act, feat_bkg, label):
        loss = {}

        label = label / torch.sum(label, dim=1, keepdim=True)

        loss_cls = self.ce_criterion(score_act, label)

        label_bkg = torch.ones_like(label)
        label_bkg /= torch.sum(label_bkg, dim=1, keepdim=True)
        loss_be = self.ce_criterion(score_bkg, label_bkg)

        loss_act = self.margin - torch.norm(torch.mean(feat_act, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_bkg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)

        loss_total = loss_cls + self.alpha * loss_um + self.beta * loss_be

        loss["loss_cls"] = loss_cls
        loss["loss_be"] = loss_be
        loss["loss_um"] = loss_um
        loss["loss_total"] = loss_total

        return loss_total, loss


class CAS_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(CAS_Module, self).__init__()
        self.len_feature = len_feature
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.drop_out = nn.Dropout(p=0.7)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)
        # out: (B, F, T)
        out = self.conv(out)
        features = out.permute(0, 2, 1)
        out = self.drop_out(out)
        out = self.classifier(out)
        out = out.permute(0, 2, 1)
        # out: (B, T, C)
        return out, features


class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act, r_bkg):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes

        self.cas_module = CAS_Module(len_feature, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.softmax_2 = nn.Softmax(dim=2)

        self.r_act = r_act
        self.r_bkg = r_bkg

        self.drop_out = nn.Dropout(p=0.7)

        # calculate loss here
        self.criterion = UM_loss(0.0005, 1, 100)

    def forward(self, x):
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act
        k_bkg = num_segments // self.r_bkg

        cas, features = self.cas_module(x)

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        
        # select_idx = torch.ones_like(feat_magnitudes).cuda()
        select_idx = torch.ones_like(feat_magnitudes)
        select_idx = self.drop_out(select_idx)

        feat_magnitudes_drop = feat_magnitudes * select_idx

        feat_magnitudes_rev = torch.max(feat_magnitudes, dim=1, keepdim=True)[0] - feat_magnitudes
        feat_magnitudes_rev_drop = feat_magnitudes_rev * select_idx

        _, sorted_idx = feat_magnitudes_drop.sort(descending=True, dim=1)
        idx_act = sorted_idx[:, :k_act]
        idx_act_feat = idx_act.unsqueeze(2).expand([-1, -1, features.shape[2]])

        _, sorted_idx = feat_magnitudes_rev_drop.sort(descending=True, dim=1)
        idx_bkg = sorted_idx[:, :k_bkg]
        idx_bkg_feat = idx_bkg.unsqueeze(2).expand([-1, -1, features.shape[2]])
        idx_bkg_cas = idx_bkg.unsqueeze(2).expand([-1, -1, cas.shape[2]])
        
        feat_act = torch.gather(features, 1, idx_act_feat)
        feat_bkg = torch.gather(features, 1, idx_bkg_feat)

        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_act, :]
        score_act = torch.mean(topk_scores, dim=1)
        score_bkg = torch.mean(torch.gather(cas, 1, idx_bkg_cas), dim=1)

        score_act = self.softmax(score_act)
        score_bkg = self.softmax(score_bkg)

        cas_softmax = self.softmax_2(cas)

        # calculate loss here
        # _label = np.zeros([20], dtype=np.float32)
        _label = torch.zeros((1, 20), dtype=torch.int64)
        cost, loss = self.criterion(score_act, score_bkg, feat_act, feat_bkg, _label)

        return score_act, score_bkg, feat_act, feat_bkg, features, cas_softmax, cost, loss


if __name__ == '__main__':
    model = Model(2048, 20, 9, 4)
    data = torch.randn((1, 750, 2048))
    # out = model(data)

    from thop import profile
    macs, params = profile(model, inputs=(data,))
    FLOPs = macs * 2
    print('FLOPs:', format(int(FLOPs), ','))  # 18,938,880,000
    print('params', format(int(params), ','))  # 12,625,920

    # from facebookresearch_params_flops_remove_g import params_count, flop_count
    #
    # print('facebookresearch params: %d' % params_count(model))  #
    # flop_dict, _ = flop_count(model, (data,))
    # flops = sum(flop_dict.values())
    # print('facebookresearch flops: %d' % flops)  # 12625920
