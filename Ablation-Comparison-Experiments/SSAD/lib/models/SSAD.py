import torch.nn as nn
import torch


class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 32, 512]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2048,
                               out_channels=256,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=256,
                               out_channels=256,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
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

        num_box = cfg.MODEL.NUM_DBOX['AL1']
        self.pred1 = nn.Conv1d(in_channels=512,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL2']
        self.pred2 = nn.Conv1d(in_channels=512,
                               out_channels=num_box * self.num_pred_value,
                               kernel_size=3, stride=1, padding=1, bias=True)
        num_box = cfg.MODEL.NUM_DBOX['AL3']
        self.pred3 = nn.Conv1d(in_channels=512,
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
        mal1, mal2, mal3 = self.main_anchor_net(base_feature)

        pred_anchor1 = self.pred1(mal1)
        pred_anchor1 = self.tensor_view(pred_anchor1)
        pred_anchor2 = self.pred2(mal2)
        pred_anchor2 = self.tensor_view(pred_anchor2)
        pred_anchor3 = self.pred3(mal3)
        pred_anchor3 = self.tensor_view(pred_anchor3)

        return pred_anchor1, pred_anchor2, pred_anchor3


if __name__ == '__main__':
    import sys
    sys.path.append('/disk3/yangle/A2Net/Ablation-Comparison/SSAD/lib')
    from config import cfg, update_config
    cfg_file = '/disk3/yangle/A2Net/Ablation-Comparison/SSAD/experiments/thumos/SSAD.yaml'
    update_config(cfg_file)

    model = SSAD(cfg)
    data = torch.randn((2, 2048, 128))
    out = model(data)
    for i in out:
        print(i.size())

