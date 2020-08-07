import torch.nn as nn
import torch
from thop import profile


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
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        fea = self.max_pooling(fea)
        return fea


class FeatNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: MAL1, MAL2, MAL3
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_feature_net = BaseFeatureNet(cfg)
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.CON1_FEAT_DIM,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.CON1_FEAT_DIM,
                               out_channels=cfg.MODEL.CON2_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.CON2_FEAT_DIM,
                               out_channels=cfg.MODEL.CON3_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=cfg.MODEL.CON3_FEAT_DIM,
                               out_channels=cfg.MODEL.CON4_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=cfg.MODEL.CON4_FEAT_DIM,
                               out_channels=cfg.MODEL.CON5_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv6 = nn.Conv1d(in_channels=cfg.MODEL.CON5_FEAT_DIM,
                               out_channels=cfg.MODEL.CON6_FEAT_DIM,
                               kernel_size=3, stride=2, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        base_feature = self.base_feature_net(x)
        mal1 = self.relu(self.conv1(base_feature))
        mal2 = self.relu(self.conv2(mal1))
        mal3 = self.relu(self.conv3(mal2))
        mal4 = self.relu(self.conv4(mal3))
        mal5 = self.relu(self.conv5(mal4))
        mal6 = self.relu(self.conv6(mal5))

        return mal1, mal2, mal3, mal4, mal5, mal6


class LocNetAB(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(LocNetAB, self).__init__()
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 3

        ##
        num_box = cfg.MODEL.NUM_DBOX['AL1']
        self.pred1 = nn.Conv1d(in_channels=cfg.MODEL.CON1_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        ##
        num_box = cfg.MODEL.NUM_DBOX['AL2']
        self.pred2 = nn.Conv1d(in_channels=cfg.MODEL.CON2_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL3']
        self.pred3 = nn.Conv1d(in_channels=cfg.MODEL.CON3_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL4']
        self.pred4 = nn.Conv1d(in_channels=cfg.MODEL.CON4_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL5']
        self.pred5 = nn.Conv1d(in_channels=cfg.MODEL.CON5_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL6']
        self.pred6 = nn.Conv1d(in_channels=cfg.MODEL.CON6_FEAT_DIM,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)

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

        mal1, mal2, mal3, mal4, mal5, mal6 = feat_list
        ##
        pred_anchor1 = self.pred1(mal1)
        pred_anchor1 = self.tensor_view(pred_anchor1)
        ##
        pred_anchor2 = self.pred2(mal2)
        pred_anchor2 = self.tensor_view(pred_anchor2)
        pred_anchor3 = self.pred3(mal3)
        pred_anchor3 = self.tensor_view(pred_anchor3)
        pred_anchor4 = self.pred4(mal4)
        pred_anchor4 = self.tensor_view(pred_anchor4)
        pred_anchor5 = self.pred5(mal5)
        pred_anchor5 = self.tensor_view(pred_anchor5)
        pred_anchor6 = self.pred6(mal6)
        pred_anchor6 = self.tensor_view(pred_anchor6)

        return pred_anchor1, pred_anchor2, pred_anchor3, pred_anchor4, pred_anchor5, pred_anchor6


############### anchor-free ##############
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
        self.pred = conv1d_c512(cfg, stride=1, out_channels=pred_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat2 = self.relu(self.conv2(feat1))
        pred = self.pred(feat2)  # [batch, 2, temporal_length]
        pred = pred.permute(0, 2, 1).contiguous()

        return pred


class PredHead(nn.Module):
    '''
    Predict classification and regression
    '''
    def __init__(self, cfg):
        super(PredHead, self).__init__()
        self.cls_branch = PredHeadBranch(cfg, pred_channels=cfg.DATASET.NUM_CLASSES)
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


class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.CON1_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.CON2_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv1d(in_channels=cfg.MODEL.CON3_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv4 = nn.Conv1d(in_channels=cfg.MODEL.CON4_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv5 = nn.Conv1d(in_channels=cfg.MODEL.CON5_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.conv6 = nn.Conv1d(in_channels=cfg.MODEL.CON6_FEAT_DIM, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_list):
        mal1, mal2, mal3, mal4, mal5, mal6 = feat_list
        mal1 = self.relu(self.conv1(mal1))
        mal2 = self.relu(self.conv2(mal2))
        mal3 = self.relu(self.conv3(mal3))
        mal4 = self.relu(self.conv4(mal4))
        mal5 = self.relu(self.conv5(mal5))
        mal6 = self.relu(self.conv6(mal6))

        return mal1, mal2, mal3, mal4, mal5, mal6


class LocNetAF(nn.Module):
    '''
    Predict action boundary, based on features from different FPN levels
    '''
    def __init__(self, cfg):
        super(LocNetAF, self).__init__()
        # self.features = FeatNet(cfg)
        self.reduce_channels = ReduceChannel(cfg)
        self.pred = PredHead(cfg)

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

        for feat, scale in zip(feat_list, scale_list):
            cls_tmp, reg_tmp = self.pred(feat)
            reg_tmp = scale(reg_tmp)
            pred_cls.append(cls_tmp)
            pred_reg.append(reg_tmp)

        predictions_cls = torch.cat(pred_cls, dim=1)
        predictions_reg = torch.cat(pred_reg, dim=1)

        return predictions_cls, predictions_reg

    def forward(self, features_list):
        features_list = self.reduce_channels(features_list)
        scale_list = [self.scale0, self.scale1, self.scale2, self.scale3, self.scale4, self.scale5, self.scale6]

        predictions_cls, predictions_reg = self._layer_cal(features_list, scale_list)

        return predictions_cls, predictions_reg


class LocNet(nn.Module):
    def __init__(self, cfg):
        super(LocNet, self).__init__()
        self.features = FeatNet(cfg)
        self.af = LocNetAF(cfg)
        self.ab = LocNetAB(cfg)

    def forward(self, x):
        features = self.features(x)
        out_af = self.af(features)
        out_ab = self.ab(features)
        return out_af, out_ab


if __name__ == '__main__':
    import sys
    import numpy as np
    sys.path.append('/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/lib')
    from config import cfg, update_config
    cfg_file = '/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/experiments/thumos/A2Net.yaml'
    update_config(cfg_file)

    model = LocNet(cfg)
    data = torch.randn((1, 2048, 128))
    macs, params = profile(model, inputs=(data,))
    FLOPs = macs * 2
    print('FLOPs: %d' % FLOPs)    # 2479209668
    print('params: %d' % params)  # 65527527


    # def params_count(model):
    #     """
    #     Compute the number of parameters.
    #     Args:
    #         model (model): model to count the number of parameters.
    #     """
    #     return np.sum([p.numel() for p in model.parameters()]).item()
    #
    #
    # print('facebookresearch params: %d' % params_count(model))
    #
    # from fvcore.nn.flop_count import flop_count
    #
    # gflop_dict, _ = flop_count(model, (data,))  #65527534
    # gflops = sum(gflop_dict.values())
    # print('facebookresearch gflops: %d' % gflops) # 1

    from facebookresearch_params_flops_remove_g import params_count, flop_count

    print('facebookresearch params: %d' % params_count(model))  #65527534
    flop_dict, _ = flop_count(model, (data,))  # 1239018496
    flops = sum(flop_dict.values())
    print('facebookresearch flops: %d' % flops)


    # Inference Time
    import time
    model = LocNet(cfg).cuda()
    data = torch.randn((1, 2048, 128)).cuda()
    t1 = time.time()
    for i in range(1000):
        res = model(data)
    t2 = time.time()
    print("Inference Time:", (t2 - t1) / 1000) # 0.013252334356307984  small change every time
