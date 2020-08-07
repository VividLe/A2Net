import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Model(torch.nn.Module):
    def __init__(self, n_feature, n_class, labels101to20=None):
        super(Model, self).__init__()
        self.labels20 = labels101to20
        self.n_class = n_class
        n_featureby2 = int(n_feature / 2)
        # FC layers for the 2 streams
        self.fc_f = nn.Linear(n_featureby2, n_featureby2)
        self.fc1_f = nn.Linear(n_featureby2, n_featureby2)
        self.fc_r = nn.Linear(n_featureby2, n_featureby2)
        self.fc1_r = nn.Linear(n_featureby2, n_featureby2)
        self.classifier_f = nn.Linear(n_featureby2, n_class)
        self.classifier_r = nn.Linear(n_featureby2, n_class)
        if n_class == 100:
            # temporal conv for activitynet
            self.conv = nn.Conv1d(n_class, n_class, kernel_size=13, stride=1, padding=12, dilation=2, bias=False,
                                  groups=n_class)

        self.apply(weights_init)
        # Params for multipliers of TCams for the 2 streams
        self.mul_r = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.mul_f = nn.Parameter(data=torch.Tensor(n_class).float().fill_(1))
        self.dropout_f = nn.Dropout(0.7)
        self.dropout_r = nn.Dropout(0.7)
        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=1)

    def forward(self, inputs, device='cpu', is_training=False, seq_len=None):
        seq_len = torch.from_numpy(
            np.array([138, 391, 286, 206, 102, 750, 385, 193, 198, 325, 318, 347, 280, 700,
                      228, 202, 147, 254, 283, 185, 219, 321, 310, 583, 62, 226, 175, 750,
                      316, 255, 89, 51]))
        # inputs - batch x seq_len x featSize
        base_x_f = inputs[:, :, 1024:]
        base_x_r = inputs[:, :, :1024]
        x_f = self.relu(self.fc_f(base_x_f))
        x_r = self.relu(self.fc_r(base_x_r))
        x_f = self.relu(self.fc1_f(x_f))
        x_r = self.relu(self.fc1_r(x_r))

        if is_training:
            x_f = self.dropout_f(x_f)
            x_r = self.dropout_r(x_r)

        cls_x_f = self.classifier_f(x_f)
        cls_x_r = self.classifier_r(x_r)

        tcam = cls_x_r * self.mul_r + cls_x_f * self.mul_f
        ### Add temporal conv for activity-net
        if self.n_class == 100:
            tcam = self.relu(self.conv(tcam.permute([0, 2, 1]))).permute([0, 2, 1])

        if seq_len is not None:
            atn = torch.zeros(0).to(device)
            mx_len = seq_len.max()
            for b in range(inputs.size(0)):
                if seq_len[b] == mx_len:
                    atnb1 = self.softmaxd1(tcam[[b], :seq_len[b]])
                else:
                    # attention over valid segments
                    atnb1 = torch.cat([self.softmaxd1(tcam[[b], :seq_len[b]]), tcam[[b], seq_len[b]:]], dim=1)
                atn = torch.cat([atn, atnb1], dim=0)
            # attention-weighted TCAM for count
            if self.labels20 is not None:
                count_feat = tcam[:, :, self.labels20] * atn[:, :, self.labels20]
            else:
                count_feat = tcam * atn

        return x_f, cls_x_f, x_r, cls_x_r, tcam, count_feat


if __name__ == '__main__':
    import sys
    import time
    from thop import profile

    # sys.path.append('/disk3/zt/code/2_TIP_rebuttal/2_A2Net/lib')
    # from config import cfg, update_config
    # cfg_file = '/disk3/zt/code/2_TIP_rebuttal/2_A2Net/experiments/thumos/A2Net.yaml'
    # update_config(cfg_file)

    # model = LocNet(cfg)
    # data = torch.randn((1, 2048, 128))

    model = Model(2048, 101, np.random.randint(20, size=20))
    data = torch.randn((1, 750, 2048))
    macs, params = profile(model, inputs=(data, 'cpu', False,torch.from_numpy(
            np.array([138, 391, 286, 206, 102, 750, 385, 193, 198, 325, 318, 347, 280, 700,
                      228, 202, 147, 254, 283, 185, 219, 321, 310, 583, 62, 226, 175, 750,
                      316, 255, 89, 51])) ))
    FLOPs = macs * 2
    print('FLOPs: %d' % FLOPs)  # 6601728000
    print('params: %d' % params) # 4405450


    from facebookresearch_params_flops_remove_g import params_count, flop_count

    print('facebookresearch params: %d' % params_count(model))  # 4405652
    flop_dict, _ = flop_count(model, (data,))  # 4401152
    flops = sum(flop_dict.values())
    print('facebookresearch flops: %d' % flops)




    model = Model(2048, 101, np.random.randint(20, size=20)).cuda()
    data = torch.randn((1, 750, 2048)).cuda()

    t1 = time.time()
    for i in range(1000):
        res = model(data, torch.device("cuda"), is_training=False, seq_len=torch.from_numpy(
            np.array([138, 391, 286, 206, 102, 750, 385, 193, 198, 325, 318, 347, 280, 700,
                      228, 202, 147, 254, 283, 185, 219, 321, 310, 583, 62, 226, 175, 750,
                      316, 255, 89, 51])).cuda())
    t2 = time.time()

    print((t2 - t1) / 1000)
    # 0.003265166759490967
    # 0.0023893194198608397
    # 0.0027587757110595704
