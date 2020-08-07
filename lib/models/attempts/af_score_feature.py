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
        self.conv1 = nn.Conv1d(in_channels=2 * cfg.MODEL.IN_FEAT_DIM,
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


class PredScoreBranch(nn.Module):
    '''
    predict score for localization confidence
    '''
    def __init__(self, cfg, in_channel=512, out_channel=1):
        super(PredScoreBranch, self).__init__()
        # 258 --> 256
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=cfg.MODEL.NET_FEAT_DIM, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = conv1d(cfg, stride=1)
        self.conv3 = conv1d(cfg, stride=1)
        self.conv4 = conv1d(cfg, stride=1)
        self.pred = conv1d(cfg, stride=1, out_channels=out_channel)
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
        # we predict confidence
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


class ActionFeatPooling(nn.Module):
    '''
    Left to right max-pooling
    '''
    def __init__(self):
        super(ActionFeatPooling, self).__init__()

    def forward(self, feat, reg):
        # feat: [batch, 256, 128], reg: [batch, 128, 2]
        assert feat.size(2) == reg.size(1)
        tem_length = feat.size(2)
        num_batch = feat.size(0)
        feature_list = list()

        for ibat in range(num_batch):
            feat_i = feat[ibat, :, :]
            reg_i = reg[ibat, :, :]

            data = list()
            for jtem in range(tem_length):
                # pool left
                reg_left = int(round(reg_i[jtem, 0].item()))
                reg_left = max(0, reg_left)
                index_left = max(jtem - reg_left, 0)
                feat_left = feat_i[:, index_left:jtem+1]
                feat_pool_left, _ = torch.max(feat_left, dim=1)
                # pool right
                reg_right = int(round(reg_i[jtem, 1].item()))
                reg_right = min(reg_right, tem_length)
                reg_right = max(reg_right, 1)
                index_right = min(jtem + reg_right, tem_length)
                feat_right = feat_i[:, jtem:index_right]
                feat_pool_right, _ = torch.max(feat_right, dim=1)
                # combine 2 features
                feat_pool = feat_pool_left + feat_pool_right
                feat_pool = torch.unsqueeze(feat_pool, dim=-1)
                data.append(feat_pool)
            feature_sig_bat = torch.cat(data, dim=1)
            feature_sig_bat = torch.unsqueeze(feature_sig_bat, dim=0)

            feature_list.append(feature_sig_bat)

        feature = torch.cat(feature_list, dim=0)
        return feature


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
        self.conv6 = conv1d(cfg, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        feat0 = self.base(x)
        feat1 = self.relu(self.conv1(feat0))
        feat2 = self.relu(self.conv2(feat1))
        feat3 = self.relu(self.conv3(feat2))
        feat4 = self.relu(self.conv4(feat3))
        feat5 = self.relu(self.conv5(feat4))
        feat6 = self.relu(self.conv6(feat5))

        # it is important in this order
        return feat0, feat1, feat2, feat3, feat4, feat5, feat6


class LocNet(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg, scale=False):
        super(LocNet, self).__init__()
        self.scale = scale
        self.features = FeatNet(cfg)
        self.pred = PredHead(cfg)
        self.action_feat_pool = ActionFeatPooling()
        self.conf = PredScoreBranch(cfg)

        if self.scale:
            self.scale0 = Scale()
            self.scale1 = Scale()
            self.scale2 = Scale()
            self.scale3 = Scale()
            self.scale4 = Scale()
            self.scale5 = Scale()
            self.scale6 = Scale()

    def _layer_cal(self, feat_list, scale_list):
        pred_cls = list()
        pred_reg = list()
        pred_conf = list()

        for feat, scale in zip(feat_list, scale_list):
            cls_tmp, reg_tmp = self.pred(feat)
            if self.scale:
                reg_tmp = scale(reg_tmp)
            # pooling feature
            feat_pooled = self.action_feat_pool(feat, reg_tmp)
            # predict confidence
            feat_cat = torch.cat((feat, feat_pooled), dim=1)
            conf_tmp = self.conf(feat_cat)

            pred_cls.append(cls_tmp)
            pred_reg.append(reg_tmp)
            pred_conf.append(conf_tmp)

        predictions_cls = torch.cat(pred_cls, dim=1)
        predictions_reg = torch.cat(pred_reg, dim=1)
        predictions_conf = torch.cat(pred_conf, dim=1)

        return predictions_cls, predictions_reg, predictions_conf

    def forward(self, x):
        feat0, feat1, feat2, feat3, feat4, feat5, feat6 = self.features(x)
        feat_list = [feat0, feat1, feat2, feat3, feat4, feat5, feat6]
        scale_list = [self.scale0, self.scale1, self.scale2, self.scale3, self.scale4, self.scale5, self.scale6]

        predictions_cls, predictions_reg, predictions_conf = self._layer_cal(feat_list, scale_list)

        return predictions_cls, predictions_reg, predictions_conf


if __name__ == '__main__':
    import sys
    sys.path.append('/data/2/v-yale/ActionLocalization/lib')
    from config import cfg, update_config
    cfg_file = '/data/2/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    model = LocNet(cfg, scale=True).cuda()
    data = torch.randn((2, 2048, 128)).cuda()
    res = model(data)
    print(res[0].size(), res[1].size(), res[2].size())


