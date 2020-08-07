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
        self.conv1 = nn.Conv1d(in_channels=303,
                               out_channels=256,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=256,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling2 = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fea1 = self.relu(self.conv1(x))
        fea1 = self.max_pooling1(fea1)
        fea = self.relu(self.conv2(fea1))
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
        self.conv1 = nn.Conv1d(in_channels=256,
                               out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3 = nn.Conv1d(in_channels=512,
                               out_channels=512,
                               kernel_size=3, stride=2, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mal1 = self.relu(self.conv1(x))
        mal2 = self.relu(self.conv2(mal1))
        mal3 = self.relu(self.conv3(mal2))

        return mal1, mal2, mal3


class SSAD(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(SSAD, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 3
        self.base_feature_net = BaseFeatureNet(cfg)
        self.main_anchor_net = MainAnchorNet(cfg)

        self.pred1 = nn.Conv1d(in_channels=512,
                               out_channels=3 * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred2 = nn.Conv1d(in_channels=512,
                               out_channels=5 * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.pred3 = nn.Conv1d(in_channels=512,
                               out_channels=5 * self.num_pred_value,
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
        mal1, mal2, mal3 = self.main_anchor_net(base_feature)

        pred_anchor1 = self.pred1(mal1)
        pred_anchor2 = self.pred2(mal2)
        pred_anchor3 = self.pred3(mal3)

        return pred_anchor1, pred_anchor2, pred_anchor3


if __name__ == '__main__':
    import sys
    sys.path.append('/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/lib')
    from config import cfg, update_config
    cfg_file = '/disk3/zt/code/2_TIP_rebuttal/5_FLOPs_inference_time/experiments/thumos/A2Net.yaml'
    update_config(cfg_file)

    model = SSAD(cfg)
    data = torch.randn((1, 303, 512))
    # out = model(data)
    # for i in out:
    #     print(i.size())
    macs, params = profile(model, inputs=(data,))
    FLOPs = macs * 2
    print('FLOPs: %d' % FLOPs)    # 905643072
    print('params: %d' % params)  # 3735608

    from facebookresearch_params_flops_remove_g import params_count, flop_count

    print('facebookresearch params: %d' % params_count(model))  # 3735608
    flop_dict, _ = flop_count(model, (data,))  # 452640768
    flops = sum(flop_dict.values())
    print('facebookresearch flops: %d' % flops)



    # Inference Time
    import time

    model = SSAD(cfg).cuda()
    data = torch.randn((1, 303, 512)).cuda()
    t1 = time.time()
    for i in range(1000):
        res = model(data)
    t2 = time.time()
    print("Inference Time:", (t2 - t1) / 1000)  # 0.00236631178855896

