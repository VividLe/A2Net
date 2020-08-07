import torch.nn as nn
import torch
import torch.nn.functional as F


def conv1d(cfg, kernel_size=3, stride=1, out_channels=None):
    if out_channels is None:
        out_channels = cfg.MODEL.NET_FEAT_DIM
    return nn.Conv1d(in_channels=cfg.MODEL.NET_FEAT_DIM, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=1, bias=True)


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


class FuseLayer(nn.Module):
    '''
    Fuse information from different scales
    '''
    def __init__(self, cfg, num_branch):
        super(FuseLayer, self).__init__()
        self.conv_down = conv1d(cfg, kernel_size=3, stride=2)
        self.num_branch = num_branch
        # TODO: inplace=True
        self.relu = nn.ReLU()

    def forward(self, x):
        output = list()
        for i in range(self.num_branch):
            fuse_in = list()
            for j in range(self.num_branch):
                data = x[j]
                # up-sample
                if j > i:
                    # print('up-sample')
                    feat_tmp = F.interpolate(data, scale_factor=2**(j-i), mode='nearest')
                # identity
                elif j == i:
                    # print('identity')
                    feat_tmp = data
                # down-sample
                else:
                    # print('down-sample')
                    for k in range(i - j):
                        data = self.relu(self.conv_down(data))
                    feat_tmp = data
                fuse_in.append(feat_tmp)

            # element-wise add
            y = fuse_in[0]
            for j in range(1, self.num_branch):
                y = y + fuse_in[j]
            output.append(y)

        return output


class StemBlock(nn.Module):
    '''
    one step forward
    '''
    def __init__(self, cfg, num_branch):
        super(StemBlock, self).__init__()
        self.num_branch = num_branch
        if self.num_branch >= 1:
            self.conv_s1 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 2:
            self.conv_s2 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 3:
            self.conv_s3 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 4:
            self.conv_s4 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 5:
            self.conv_s5 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 6:
            self.conv_s6 = conv1d(cfg, kernel_size=3, stride=1)
        if self.num_branch >= 7:
            self.conv_s7 = conv1d(cfg, kernel_size=3, stride=1)
        # TODO: inplace=True
        self.relu = nn.ReLU()

    def forward(self, x):
        output = list()
        if self.num_branch >= 1:
            out_tmp = self.relu(self.conv_s1(x[0]))
            output.append(out_tmp)
        if self.num_branch >= 2:
            out_tmp = self.relu(self.conv_s2(x[1]))
            output.append(out_tmp)
        if self.num_branch >= 3:
            out_tmp = self.relu(self.conv_s3(x[2]))
            output.append(out_tmp)
        if self.num_branch >= 4:
            out_tmp = self.relu(self.conv_s4(x[3]))
            output.append(out_tmp)
        if self.num_branch >= 5:
            out_tmp = self.relu(self.conv_s5(x[4]))
            output.append(out_tmp)
        if self.num_branch >= 6:
            out_tmp = self.relu(self.conv_s6(x[5]))
            output.append(out_tmp)
        if self.num_branch >= 7:
            out_tmp = self.relu(self.conv_s7(x[6]))
            output.append(out_tmp)
        return output


class FeatNet(nn.Module):
    '''
    Action localization with high resolution representation
    '''
    def __init__(self, cfg):
        super(FeatNet, self).__init__()
        self.base_layer = BaseFeatureNet(cfg)
        self.stem1 = StemBlock(cfg, num_branch=1)
        self.down1 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem2 = StemBlock(cfg, num_branch=2)
        self.fuse2 = FuseLayer(cfg, num_branch=2)
        self.down2 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem3 = StemBlock(cfg, num_branch=3)
        self.fuse3 = FuseLayer(cfg, num_branch=3)
        self.down3 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem4 = StemBlock(cfg, num_branch=4)
        self.fuse4 = FuseLayer(cfg, num_branch=4)
        self.down4 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem5 = StemBlock(cfg, num_branch=5)
        self.fuse5 = FuseLayer(cfg, num_branch=5)
        self.down5 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem6 = StemBlock(cfg, num_branch=6)
        self.fuse6 = FuseLayer(cfg, num_branch=6)
        self.down6 = conv1d(cfg, kernel_size=3, stride=2)

        self.stem7 = StemBlock(cfg, num_branch=7)
        self.fuse7 = FuseLayer(cfg, num_branch=7)

        # TODO: inplace
        self.relu = nn.ReLU()

    def forward(self, x):
        feat = self.base_layer(x)
        feats1_in = [feat]
        feats1_out = self.stem1(feats1_in)
        feat_down1 = self.relu(self.down1(feat))
        feats1_out.append(feat_down1)

        feats2_out = self.stem2(feats1_out)
        feat_down2 = self.relu(self.down2(feats2_out[-1]))
        feats2_out = self.fuse2(feats2_out)
        feats2_out.append(feat_down2)

        feats3_out = self.stem3(feats2_out)
        feat_down3 = self.relu(self.down3(feats3_out[-1]))
        feats3_out = self.fuse3(feats3_out)
        feats3_out.append(feat_down3)

        feats4_out = self.stem4(feats3_out)
        feat_down4 = self.relu(self.down4(feats4_out[-1]))
        feats4_out = self.fuse4(feats4_out)
        feats4_out.append(feat_down4)

        feats5_out = self.stem5(feats4_out)
        feat_down5 = self.relu(self.down5(feats5_out[-1]))
        feats5_out = self.fuse5(feats5_out)
        feats5_out.append(feat_down5)

        feats6_out = self.stem6(feats5_out)
        feat_down6 = self.relu(self.down6(feats6_out[-1]))
        feats6_out = self.fuse6(feats6_out)
        feats6_out.append(feat_down6)

        feats7_out = self.stem7(feats6_out)
        feats7_out = self.fuse7(feats7_out)

        return feats7_out


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
            self.scale6 = Scale()

    def forward(self, x):
        feat0, feat1, feat2, feat3, feat4, feat5, feat6 = self.features(x)

        pred_cls = list()
        pred_reg = list()

        cls_tmp, reg_tmp = self.pred(feat0)
        if self.scale:
            reg_tmp = self.scale0(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        cls_tmp, reg_tmp = self.pred(feat1)
        if self.scale:
            reg_tmp = self.scale1(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        cls_tmp, reg_tmp = self.pred(feat2)
        if self.scale:
            reg_tmp = self.scale2(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        cls_tmp, reg_tmp = self.pred(feat3)
        if self.scale:
            reg_tmp = self.scale3(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        cls_tmp, reg_tmp = self.pred(feat4)
        if self.scale:
            reg_tmp = self.scale4(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

        cls_tmp, reg_tmp = self.pred(feat5)
        if self.scale:
            reg_tmp = self.scale5(reg_tmp)
        pred_cls.append(cls_tmp)
        pred_reg.append(reg_tmp)

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
    import sys
    sys.path.append('/data/2/v-yale/ActionLocalization/lib')
    from config import cfg, update_config
    cfg_file = '/data/2/v-yale/ActionLocalization/experiments/anet/ssad.yaml'
    update_config(cfg_file)

    model = LocNet(cfg, scale=True).cuda()
    data = torch.randn((2, 2048, 128)).cuda()
    res = model(data)
    print(res[0].size(), res[1].size())

    # data1 = torch.randn((2, 1024, 128)).cuda()
    # net = FeatNet(cfg).cuda()
    # output = net(data1)
    # for iord, out in enumerate(output):
    #     print(iord, out.size())

    # data1 = torch.randn((2, 256, 128)).cuda()
    # data2 = torch.randn((2, 256, 64)).cuda()
    # data3 = torch.randn((2, 256, 32)).cuda()
    # data4 = torch.randn((2, 256, 16)).cuda()
    # data5 = torch.randn((2, 256, 8)).cuda()
    # data6 = torch.randn((2, 256, 4)).cuda()
    # data7 = torch.randn((2, 256, 2)).cuda()
    #
    # # net = FuseLayer(cfg).cuda()
    #
    # x = [data1, data2, data3, data4, data5, data6, data7]
    # num_branch = len(x)
    # net = StemBlock(cfg, num_branch=num_branch).cuda()
    # output = net(x)
    # for iord, out in enumerate(output):
    #     print(iord, out.size())

