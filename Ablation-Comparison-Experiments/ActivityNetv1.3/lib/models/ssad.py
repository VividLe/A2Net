import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

def conv1d(cfg, stride, in_channels, out_channels, ):
    # if in_channels is None:
    #     in_channels = cfg.MODEL.NET_FEAT_DIM
    # if out_channels is None:
    #     out_channels = cfg.MODEL.NET_FEAT_DIM
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=True)
def conv1d_c512(cfg, stride, out_channels=None):
    if out_channels is None:
        out_channels = 512
    return nn.Conv1d(in_channels=512, out_channels=out_channels,
                     kernel_size=3, stride=stride, padding=1, bias=True)



class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 32, 512]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2048,
                               out_channels=512,
                               kernel_size=1, stride=1, bias=True)
        # self.max_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        # fea = self.max_pooling1(fea)
        fea = self.relu(self.conv2(fea))
        out = self.max_pooling(fea)
        return out


class FeatNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: feat1, feat2, feat3
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base = BaseFeatureNet(cfg)
        self.conv1 = conv1d(cfg, stride=1, in_channels=512, out_channels=512)
        self.conv2 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.conv3 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.conv4 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.conv5 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.conv6 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.conv7 = conv1d(cfg, stride=2, in_channels=512, out_channels=512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat0 = self.base(x)
        feat1 = self.relu(self.conv1(feat0))
        feat2 = self.relu(self.conv2(feat1))
        feat3 = self.relu(self.conv3(feat2))
        feat4 = self.relu(self.conv4(feat3))
        feat5 = self.relu(self.conv5(feat4))
        feat6 = self.relu(self.conv6(feat5))
        feat7 = self.relu(self.conv7(feat6))

        return feat1, feat2, feat3, feat4, feat5, feat6, feat7


class LocNetAB(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(LocNetAB, self).__init__()
        # self.batch_size = cfg.TRAIN.BATCH_SIZE
        # only predict overlap
        self.num_pred_value = 1
        # self.base_feature_net = BaseFeatureNet(cfg)
        # self.main_anchor_net = FeatNet(cfg)
        self.sigmoid = nn.Sigmoid()

        num_box = cfg.MODEL.NUM_DBOX['AL0']
        self.pred1 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL1']
        self.pred2 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL2']
        self.pred3 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL3']
        self.pred4 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL4']
        self.pred5 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL5']
        self.pred6 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)
        num_box = cfg.MODEL.NUM_DBOX['AL6']
        self.pred7 = conv1d(cfg, stride=1, in_channels=512, out_channels=num_box * self.num_pred_value)

    def tensor_view(self, data):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        batch, cha, dep = data.size()
        data = data.view(batch, -1, self.num_pred_value, dep)
        data = data.permute(0, 3, 1, 2).contiguous()
        data = data.view(batch, -1, self.num_pred_value)
        return data

    def forward(self, feat_list):
        # base_feature = self.base_feature_net(x)
        # feat1, feat2, feat3, feat4, feat5, feat6, feat7 = self.main_anchor_net(x)


        # feat1, feat2, feat3, feat4, feat5, feat6, feat7 = feat_list
        # pred_anchor1 = self.pred1(feat1)
        # pred_anchor1 = self.sigmoid(self.tensor_view(pred_anchor1))
        # pred_anchor2 = self.pred2(feat2)
        # pred_anchor2 = self.sigmoid(self.tensor_view(pred_anchor2))
        # pred_anchor3 = self.pred3(feat3)
        # pred_anchor3 = self.sigmoid(self.tensor_view(pred_anchor3))
        # pred_anchor4 = self.pred4(feat4)
        # pred_anchor4 = self.sigmoid(self.tensor_view(pred_anchor4))
        # pred_anchor5 = self.pred5(feat5)
        # pred_anchor5 = self.sigmoid(self.tensor_view(pred_anchor5))
        # pred_anchor6 = self.pred6(feat6)
        # pred_anchor6 = self.sigmoid(self.tensor_view(pred_anchor6))
        # pred_anchor7 = self.pred7(feat7)
        # pred_anchor7 = self.sigmoid(self.tensor_view(pred_anchor7))
        feat1, feat2, feat3, feat4, feat5, feat6, feat7 = feat_list
        pred_anchor1 = self.pred1(feat1)
        pred_anchor1 = self.tensor_view(pred_anchor1)
        pred_anchor2 = self.pred2(feat2)
        pred_anchor2 = self.tensor_view(pred_anchor2)
        pred_anchor3 = self.pred3(feat3)
        pred_anchor3 = self.tensor_view(pred_anchor3)
        pred_anchor4 = self.pred4(feat4)
        pred_anchor4 = self.tensor_view(pred_anchor4)
        pred_anchor5 = self.pred5(feat5)
        pred_anchor5 = self.tensor_view(pred_anchor5)
        pred_anchor6 = self.pred6(feat6)
        pred_anchor6 = self.tensor_view(pred_anchor6)
        pred_anchor7 = self.pred7(feat7)
        pred_anchor7 = self.tensor_view(pred_anchor7)
        return pred_anchor1, pred_anchor2, pred_anchor3, pred_anchor4, pred_anchor5, pred_anchor6, pred_anchor7


# class UnityChannel(nn.Module):
#     def __init__(self, cfg):
#         super(UnityChannel, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.conv3 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.conv4 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.conv5 = nn.Conv1d(in_channels=2048, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.conv6 = nn.Conv1d(in_channels=4096, out_channels=512, kernel_size=1, padding=0, bias=True)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, feat_list):
#         feat1, feat2, feat3, feat4, feat5, feat6, feat7 = feat_list
#         # to do af pre at 1
#         feat1 = self.relu(self.conv1(feat1))
#         feat2 = self.relu(self.conv2(feat2))
#         feat3 = self.relu(self.conv3(feat3))
#         feat4 = self.relu(self.conv4(feat4))
#         feat5 = self.relu(self.conv5(feat5))
#         feat6 = self.relu(self.conv6(feat6))
#         return feat1, feat2, feat3, feat4, feat5, feat6


class PredHeadBranch(nn.Module):
    '''
    prediction module branch
    Output number is 2
      Regression: distance to left boundary, distance to right boundary
      Classification: probability
    '''
    def __init__(self, cfg, pred_channels=2):
        super(PredHeadBranch, self).__init__()
        self.conv1 = conv1d_c512(cfg, stride=1)
        self.conv2 = conv1d_c512(cfg, stride=1)
        self.conv3 = conv1d_c512(cfg, stride=1)
        self.conv4 = conv1d_c512(cfg, stride=1)
        self.pred = conv1d_c512(cfg, stride=1, out_channels=pred_channels)
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


class LocNetAF(nn.Module):
    def __init__(self, cfg, scale= True):
        super(LocNetAF, self).__init__()
        # self.unity_channel = UnityChannel(cfg)
        self.scale = scale
        self.pred = PredHead(cfg)
        if self.scale:
            # self.scale0 = Scale()
            self.scale1 = Scale()
            self.scale2 = Scale()
            self.scale3 = Scale()
            self.scale4 = Scale()
            self.scale5 = Scale()
            self.scale6 = Scale()

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

    def forward(self, feat_list):

        # features_list = self.unity_channel(feat_list)
        scale_list = [ self.scale1, self.scale2, self.scale3, self.scale4, self.scale5, self.scale6]

        predictions_cls, predictions_reg = self._layer_cal(feat_list, scale_list)

        return predictions_cls, predictions_reg


class LocNet(nn.Module):
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        self.features = FeatNet(cfg)
        self.ab = LocNetAB(cfg)
        self.af = LocNetAF(cfg)
    def forward(self, x, is_ab):
        features = self.features(x)
        if is_ab:
            outputs = self.ab(features)
        else:
            outputs = self.af(features)
        return outputs 


if __name__ == '__main__':
    sys.path.append('/data/yangle/zt/ab_af_new_zt/lib')
    from config import cfg, update_config
    cfg_file = '/data/yangle/zt/ab_af_new_zt/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    data = torch.randn((2, 2048, 128)).cuda()

    model = LocNet(cfg).cuda()

    #ab
    predictions = model(data, is_ab = True)
    for i, pred in enumerate(predictions):
        print(i, pred.size())

    # res = model(data, is_ab = False)
    # print(res[0].size(), res[1].size())