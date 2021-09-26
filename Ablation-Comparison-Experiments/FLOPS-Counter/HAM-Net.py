'''
A Hybrid Attention Mechanism for Weakly-Supervised Temporal Action Localization
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ADL(nn.Module):
    def __init__(self, drop_thres=0.5, drop_prob=0.5):
        super().__init__()
        self.drop_thres = drop_thres
        self.drop_prob = drop_prob

    def forward(self, x, x_atn, include_min=False):
        if not self.training:
            return x_atn, x_atn

        # important mask
        mask_imp = x_atn

        # drop mask
        if include_min:
            atn_max = x_atn.max(dim=-1, keepdim=True)[0]
            atn_min = x_atn.min(dim=-1, keepdim=True)[0]
            _thres = (atn_max - atn_min) * self.drop_thres + atn_min
        else:
            _thres = x_atn.max(dim=-1, keepdim=True)[0] * self.drop_thres
        drop_mask = (x_atn < _thres).type_as(x) * x_atn

        return mask_imp, drop_mask


class HAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_class = 20
        n_feature = 2048

        self.classifier = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(n_feature, n_feature, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Dropout(0.7), nn.Conv1d(n_feature, n_class + 1, 1))

        self.attention = nn.Sequential(nn.Conv1d(n_feature, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Sigmoid())

        self.adl = ADL(drop_thres=0.5, drop_prob=0.5)

    def forward(self, inputs, include_min=False):
        x = inputs.transpose(-1, -2)
        x_cls = self.classifier(x)
        x_atn = self.attention(x)

        atn_supp, atn_drop = self.adl(x_cls, x_atn, include_min=include_min)

        return x_cls.transpose(-1, -2), atn_supp.transpose(
            -1, -2), atn_drop.transpose(-1, -2), x_atn.transpose(-1, -2)


if __name__ == '__main__':
    model = HAMNet()
    data = torch.randn((1, 750, 2048))
    out = model(data)

    from thop import profile

    macs, params = profile(model, inputs=(data,))
    FLOPs = macs * 2
    print('FLOPs:', format(int(FLOPs), ','))  # 43,727,649,000
    print('params', format(int(params), ','))  # 29,146,646
