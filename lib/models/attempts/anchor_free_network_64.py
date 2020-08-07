import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch.nn as nn
import torch
import torch.nn.functional as F


def conv1d(cfg, stride, out_channels=None):
    if out_channels is None:
        out_channels = cfg.MODEL.NET_FEAT_DIM
    return nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=True)


class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 128, 256]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.NET_FEAT_DIM,
                               kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM,
                               out_channels=cfg.MODEL.NET_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        out = self.relu(self.conv2(feat1))
        return out


class PredHeadBranch(nn.Module):
    '''
    prediction module branch
    Output number is 2
      Regression: distance to left boundary, distance to right boundary
      Classification: probability
    '''
    def __init__(self, cfg, pred_channels=2):
        super(PredHeadBranch, self).__init__()
        self.conv1 = conv1d(cfg, stride=1)
        self.conv2 = conv1d(cfg, stride=1)
        self.conv3 = conv1d(cfg, stride=1)
        self.conv4 = conv1d(cfg, stride=1)
        self.pred = conv1d(cfg, stride=1, out_channels=pred_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.relu(self.conv2(feat1))
        feat3 = self.relu(self.conv3(feat2))
        feat4 = self.relu(self.conv4(feat3))
        pred = self.pred(feat4)  # [batch, 2, temporal_length]
        pred = pred.permute(0, 2, 1).contiguous()

        return pred


class PredHead(nn.Module):
    '''
    Predict classification and regression
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.cls_branch = PredHeadBranch(cfg, pred_channels=2)
        self.reg_branch = PredHeadBranch(cfg)

    def forward(self, x):
        # predict the probability for foreground or background
        cls = self.cls_branch(x)
        reg = self.reg_branch(x)

        return cls, reg


class Scale(nn.Module):
    '''
    Different layers regression to different size range
    Learn a trainable scalar to automatically adjust the base of exp(si * x)
    '''
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FeatNet(nn.Module):
    '''
    Action localization Network
    128 --> 64 --> 32 --> 16 --> 8 --> 4 --> 2
    FPN architecture, 2 --> 4 --> 8 --> 16 --> 32 --> 64 --> 128
    input: action feature, [batch_size, 1024, 128]
    output:
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        # down-sample path
        self.base = BaseFeatureNet(cfg)
        self.conv1 = conv1d(cfg, stride=2)
        self.conv2 = conv1d(cfg, stride=2)
        self.conv3 = conv1d(cfg, stride=2)
        self.conv4 = conv1d(cfg, stride=2)
        self.conv5 = conv1d(cfg, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        feat0 = self.base(x)
        feat1 = self.relu(self.conv1(feat0))
        feat2 = self.relu(self.conv2(feat1))
        feat3 = self.relu(self.conv3(feat2))
        feat4 = self.relu(self.conv4(feat3))
        feat5 = self.relu(self.conv5(feat4))

        # it is important in this order
        return feat0, feat1, feat2, feat3, feat4, feat5


class LocNet(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg, scale=False):
        super(LocNet, self).__init__()
        self.scale = scale
        self.features = FeatNet(cfg)
        self.pred = PredHead(cfg)

        if self.scale:
            self.scale0 = Scale()
            self.scale1 = Scale()
            self.scale2 = Scale()
            self.scale3 = Scale()
            self.scale4 = Scale()
            self.scale5 = Scale()

    def _layer_cal(self, feat_list, scale_list):
        pred_cls = list()
        pred_reg = list()

        for feat, scale in zip(feat_list, scale_list):
            cls_tmp, reg_tmp = self.pred(feat)
            if self.scale:
                reg_tmp = scale(reg_tmp)
            pred_cls.append(cls_tmp)
            pred_reg.append(reg_tmp)

        predictions_cls = torch.cat(pred_cls, dim=1)
        predictions_reg = torch.cat(pred_reg, dim=1)

        return predictions_cls, predictions_reg

    def forward(self, x):
        features_list = self.features(x)
        scale_list = [self.scale0, self.scale1, self.scale2, self.scale3, self.scale4, self.scale5]

        predictions_cls, predictions_reg = self._layer_cal(features_list, scale_list)

        return predictions_cls, predictions_reg


if __name__ == '__main__':
    import sys
    sys.path.append('/disk2/zt/code/actloc_fixed_length_github/lib')
    from config import cfg, update_config
    cfg_file = '/disk2/zt/code/actloc_fixed_length_github/experiments/anet/af.yaml'
    update_config(cfg_file)

    model = LocNet(cfg, scale=True).cuda()
    data = torch.randn((2, 2048, 64)).cuda()
    res = model(data)
    print(res[0].size(), res[1].size())
