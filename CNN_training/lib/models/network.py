# ------------------------------------------------------------------------------
# Author: Tao Zhao
# Descriptions:
# input & output: [batch, channel, temporal_length]
# cfg.NETWORK.FEAT_DIM: feature dimention, 1024 for rgb or flow, 2048 for rgb concatenate flow
# cfg.DATASET.CLS_NUM: numbers of class, 20 for THUMOS14, 200 for AcyivityNet
# cfg.NETWORK.CASMODULE_DROPOUT: probability of an element to be zeroed,
#                                used in CASModule Dropout layer, 0.7 for THUMOS14
# cfg.DATASET.NUM_SEGMENTS // cfg.NETWORK.TOPK_K_R: parameters in torch.topk,
#                                                   NUM_SEGMENTS is 750 for each video
#                                                   cfg.NETWORK.TOPK_K_R, 8 for THUMOS14
#
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init


# todo: implement weight init
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()


# cls
class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        # Notice: I modify the kernel_size from 32 to 33
        # self.conv_1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=33,
        #                         stride=1, padding=16)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.conv_2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=33,
        #                         stride=1, padding=16)
        # #self.bn2 = nn.BatchNorm1d(64)
        # self.conv_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=33,
        #                         stride=1, padding=16)
        # #self.bn3 = nn.BatchNorm1d(32)

        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=33,
                                stride=1, padding=16)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=33,
                                stride=1, padding=16)
        self.dropout = nn.Dropout(p=0.2)
        self.pred = nn.Conv1d(in_channels=64, out_channels=2, kernel_size=33,
                                stride=1, padding=16)
        self.relu = nn.ReLU()
        # top-k
        self.topk = int(cfg.DATASET.NUM_POINTS * cfg.NETWORK.TOPK_K_R)

    def forward(self, x):
        feat1 = self.relu(self.conv_1(x))
        feat2 = self.relu(self.conv_2(feat1))
        #feat3 = self.relu(self.conv_2(feat2))
        # feat2_n = feat2.cpu().detach().numpy()
        feat = self.dropout(feat2)
        # feat_n = feat.cpu().detach().numpy()
        cas = self.pred(feat)  # [b, 2, 405]
        # apply top-k means to obtain classification score
        values, _ = torch.topk(cas, k=self.topk, dim=2)  # [b, 2, topk]
        prediction = torch.mean(values, dim=2)

        return prediction


# # original
# class Network(nn.Module):
#     def __init__(self, cfg):
#         super(Network, self).__init__()
#         # Notice: I modify the kernel_size from 32 to 33
#         self.conv_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=33,
#                                 stride=1, padding=16)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm1d(num_features=64)
#
#         self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.tanh = nn.Tanh()
#
#         self.dropout = nn.Dropout(p=0.2)
#         self.linear = nn.Linear(in_features=64, out_features=2)
#
#     def forward(self, x):
#
#         # todo: the usual order is conv-bn-relu, this is quite strange
#         feature = self.conv_1(x)
#         out_conv = self.relu(feature)
#         out_conv = self.bn(out_conv)
#
#         # first max-pooling, then mean, actually use top 50% for prediction
#         out_maxp = self.max_pool(out_conv)
#         # implement global average pooling
#         out_avgp = torch.mean(out_maxp, dim=2)
#         # a bit un-reasonable to use tanh
#         out_avgp = self.tanh(out_avgp)
#
#         out_drop = self.dropout(out_avgp)
#         prediction = self.linear(out_drop)
#
#         return prediction


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/yangle/qiyu/lib')
    from config.default import config as cfg
    from config.default import update_config
    # from core.functions import weight_init

    cfg_file = '/home/yangle/qiyu/experiments/cls.yaml'
    update_config(cfg_file)

    data = torch.randn((2, 1, 405)).cuda()
    network = Network(cfg).cuda()
    prediction = network(data)

