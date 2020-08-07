import torch.nn as nn
import torch
from thop import profile


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
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.max_pooling2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea = self.relu(self.conv1(x))
        fea = self.relu(self.conv2(fea))
        out = self.max_pooling2(fea)
        return out


class MainAnchorNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: MAL1, MAL2, MAL3
    '''
    def __init__(self, cfg):
        super(MainAnchorNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=512,
                               out_channels=1024,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv1d(in_channels=1024,
                               out_channels=1024,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5 = nn.Conv1d(in_channels=1024,
                               out_channels=2048,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv6 = nn.Conv1d(in_channels=2048,
                               out_channels=2048,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv7 = nn.Conv1d(in_channels=2048,
                               out_channels=4096,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv8 = nn.Conv1d(in_channels=4096,
                               out_channels=4096,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mal1 = self.relu(self.conv1(x))
        mal2 = self.relu(self.conv2(mal1))
        mal3 = self.relu(self.conv3(mal2))
        mal4 = self.relu(self.conv4(mal3))
        mal5 = self.relu(self.conv5(mal4))
        mal6 = self.relu(self.conv6(mal5))
        mal7 = self.relu(self.conv7(mal6))
        mal8 = self.relu(self.conv8(mal7))

        return mal1, mal2, mal3, mal4, mal5, mal6, mal7, mal8


class GTAN(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(GTAN, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 3
        self.base_feature_net = BaseFeatureNet(cfg)
        self.main_anchor_net = MainAnchorNet(cfg)

        num_box = 3
        self.pred1 = nn.Conv1d(in_channels=512,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv1d(in_channels=512,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv1d(in_channels=1024,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred4 = nn.Conv1d(in_channels=1024,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred5 = nn.Conv1d(in_channels=2048,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred6 = nn.Conv1d(in_channels=2048,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=1, stride=1, bias=True)
        self.pred7 = nn.Conv1d(in_channels=4096,
                              out_channels=num_box * self.num_pred_value,
                              kernel_size=3, stride=1, padding=1, bias=True)
        self.pred8 = nn.Conv1d(in_channels=4096,
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

    def forward(self, x):
        base_feature = self.base_feature_net(x)
        mal1, mal2, mal3, mal4, mal5, mal6, mal7, mal8 = self.main_anchor_net(base_feature)

        pred_anchor1 = self.pred1(mal1)
        pred_anchor2 = self.pred2(mal2)
        pred_anchor3 = self.pred3(mal3)
        pred_anchor4 = self.pred4(mal4)
        pred_anchor5 = self.pred5(mal5)
        pred_anchor6 = self.pred6(mal6)
        pred_anchor7 = self.pred7(mal7)
        pred_anchor8 = self.pred8(mal8)

        return pred_anchor1, pred_anchor2, pred_anchor3, pred_anchor4, pred_anchor5, pred_anchor6, pred_anchor7, pred_anchor8


if __name__ == '__main__':
    import sys
    sys.path.append('/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/lib')
    from config import cfg, update_config
    cfg_file = '/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/experiments/thumos/A2Net.yaml'
    update_config(cfg_file)

    model = GTAN(cfg)
    data = torch.randn((1, 2048, 512))
    macs, params = profile(model, inputs=(data,))
    FLOPs = macs * 2
    print('FLOPs: %d' % FLOPs)    # 5011222384
    print('params: %d' % params)  # 107635264

    from facebookresearch_params_flops_remove_g import params_count, flop_count

    print('facebookresearch params: %d' % params_count(model))  # 107635264
    flop_dict, _ = flop_count(model, (data,))  # 2504884224
    flops = sum(flop_dict.values())
    print('facebookresearch flops: %d' % flops)

    # Inference Time
    import time
    model = GTAN(cfg).cuda()
    data = torch.randn((1, 2048, 512)).cuda()
    t1 = time.time()
    for i in range(1000):
        res = model(data)
    t2 = time.time()
    print("Inference Time:", (t2 - t1) / 1000)  # 0.01113538670539856
