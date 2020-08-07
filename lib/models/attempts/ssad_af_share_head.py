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
        # following paper FCOS, employ exp(x) to map real number to (0, +infi) for the regression
        # however, from experiments on mini data, this converge to large loss ~1.0
        # if we remove exp(x), training loss can converge to 0.00*
        # reg = torch.exp(reg)
        return cls, reg


# TODO: why no relu after conv?
class NonLocal(nn.Module):
    '''
    Non local module to model pair-wise connection
    '''
    def __init__(self, cfg):
        super(NonLocal, self).__init__()
        self.theta = nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM,
                                    out_channels=cfg.MODEL.NET_FEAT_DIM, kernel_size=1)
        self.phi = nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM,
                                    out_channels=cfg.MODEL.NET_FEAT_DIM, kernel_size=1)
        # TODO: I can remove self.g and self.w
        self.g = nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM,
                                    out_channels=cfg.MODEL.NET_FEAT_DIM, kernel_size=1)
        self.w = nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM,
                                     out_channels=cfg.MODEL.NET_FEAT_DIM, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # feat_theta = self.theta(x)
        feat_theta = self.relu(self.theta(x))
        feat_theta = feat_theta.permute(0, 2, 1)
        # feat_phi = self.phi(x)
        feat_phi = self.relu(self.phi(x))
        feat = torch.matmul(feat_theta, feat_phi)  # [batch, temporal_length, temporal_length]
        feat_div_c = F.softmax(feat, dim=2)

        # feat_g = self.g(x)
        feat_g = self.relu(self.g(x))
        feat_g = feat_g.permute(0, 2, 1)

        feature = torch.matmul(feat_div_c, feat_g)
        feature = feature.permute(0, 2, 1).contiguous()

        # w_feature = self.w(feature)
        w_feature = self.relu(self.w(feature))

        out = w_feature + x
        return out


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


class LocNet(nn.Module):
    '''
    Action localization Network, 7 layers
    128 --> 64 --> 32 --> 16 --> 8 --> 4 --> 2
    input: action feature, [batch_size, 1024, 128]
    output:
    '''
    def __init__(self, cfg, non_local=False, scale=False):
        super(LocNet, self).__init__()
        self.non_local = non_local
        self.scale = scale
        self.base = BaseFeatureNet(cfg)
        self.conv1 = conv1d(cfg, stride=2)
        self.conv2 = conv1d(cfg, stride=2)
        self.conv3 = conv1d(cfg, stride=2)
        self.conv4 = conv1d(cfg, stride=2)
        self.conv5 = conv1d(cfg, stride=2)
        self.conv6 = conv1d(cfg, stride=2)

        self.pred = PredHead(cfg)
        self.relu = nn.ReLU(inplace=True)

        if self.non_local:
            self.nl0 = NonLocal(cfg)
            self.nl1 = NonLocal(cfg)
            self.nl2 = NonLocal(cfg)
            self.nl3 = NonLocal(cfg)
            self.nl4 = NonLocal(cfg)
            self.nl5 = NonLocal(cfg)
            self.nl6 = NonLocal(cfg)

        if self.scale:
            self.scale0 = Scale()
            self.scale1 = Scale()
            self.scale2 = Scale()
            self.scale3 = Scale()
            self.scale4 = Scale()
            self.scale5 = Scale()
            self.scale6 = Scale()

    def forward(self, x):
        pred_cls = list()
        pred_reg = list()

        feat0 = self.base(x)
        if self.non_local:
            feat0 = self.nl0(feat0)
        cls_tmp, reg_tmp = self.pred(feat0)
        if self.scale:
            reg_tmp = self.scale0(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat1 = self.relu(self.conv1(feat0))
        if self.non_local:
            feat1 = self.nl1(feat1)
        cls_tmp, reg_tmp = self.pred(feat1)
        if self.scale:
            reg_tmp = self.scale1(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat2 = self.relu(self.conv2(feat1))
        if self.non_local:
            feat2 = self.nl2(feat2)
        cls_tmp, reg_tmp = self.pred(feat2)
        if self.scale:
            reg_tmp = self.scale2(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat3 = self.relu(self.conv3(feat2))
        if self.non_local:
            feat3 = self.nl3(feat3)
        cls_tmp, reg_tmp = self.pred(feat3)
        if self.scale:
            reg_tmp = self.scale3(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat4 = self.relu(self.conv4(feat3))
        if self.non_local:
            feat4 = self.nl4(feat4)
        cls_tmp, reg_tmp = self.pred(feat4)
        if self.scale:
            reg_tmp = self.scale4(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat5 = self.relu(self.conv5(feat4))
        if self.non_local:
            feat5 = self.nl5(feat5)
        cls_tmp, reg_tmp = self.pred(feat5)
        if self.scale:
            reg_tmp = self.scale5(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        feat6 = self.relu(self.conv6(feat5))
        if self.non_local:
            feat6 = self.nl6(feat6)
        cls_tmp, reg_tmp = self.pred(feat6)
        if self.scale:
            reg_tmp = self.scale6(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        # collect predictions
        predictions_cls = torch.cat(pred_cls, dim=1)
        predictions_reg = torch.cat(pred_reg, dim=1)

        return predictions_cls, predictions_reg


if __name__ == '__main__':
    from config import cfg, update_config
    cfg_file = '/data/2/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    model = LocNet(cfg, non_local=False, scale=True).cuda()
    data = torch.randn((2, 1024, 128)).cuda()
    res = model(data)
    print(res[0].size(), res[1].size())

