# ------------------------------------------------------------------------------
# Author: Tao Zhao
# Descriptions:
# cfg.TRAIN.C_LOSS_NORM: coefficient for loss_norm, 0.0001 for THUMOS14
# note: fore_weights is [batch, channel, temporal_length] , different from original BaSNet, so dim=2 in
#       loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=2))
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


# class BasNetLoss(nn.Module):
#     def __init__(self):
#         super(BasNetLoss, self).__init__()
#         self.ce_criterion = nn.BCELoss()
#
#     def forward(self, score_base, score_supp, fore_weights, label, cfg):
#         loss_dict = {}
#
#         label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)
#         label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1)
#
#         label_base = label_base / torch.sum(label_base, dim=1, keepdim=True)
#         label_supp = label_supp / torch.sum(label_supp, dim=1, keepdim=True)
#
#         loss_base = self.ce_criterion(score_base, label_base)
#         loss_supp = self.ce_criterion(score_supp, label_supp)
#         loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=2))
#
#         loss_total = loss_base + loss_supp + cfg.TRAIN.C_LOSS_NORM * loss_norm
#
#         loss_dict["loss_base"] = loss_base
#         loss_dict["loss_supp"] = loss_supp
#         loss_dict["loss_norm"] = loss_norm
#         loss_dict["loss_total"] = loss_total
#
#         return loss_total, loss_dict


class BasNetLoss(nn.Module):
    def __init__(self):
        super(BasNetLoss, self).__init__()

    def _cls_loss(self, cfg, labels, logits):
        '''
        calculate classification loss
        1. dispose label, ensure the sum is 1
        2. calculate topk mean, indicates classification score
        3. calculate loss
        '''
        k = int(np.ceil(cfg.DATASET.NUM_SEGMENTS / 8))
        labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-10)
        logits = torch.mean(torch.topk(logits, k, dim=2)[0], dim=2)
        # todo: we use log_softmax to calculate loss, pay attention to evaluation process
        clsloss = -torch.mean(torch.sum(labels * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return clsloss

    def forward(self, score_base, score_supp, fore_weights, label, cfg):
        loss_dict = {}

        label_base = torch.cat((label, torch.ones((label.shape[0], 1)).cuda()), dim=1)
        label_supp = torch.cat((label, torch.zeros((label.shape[0], 1)).cuda()), dim=1)

        loss_base = self._cls_loss(cfg, label_base, score_base)
        loss_supp = self._cls_loss(cfg, label_supp, score_supp)
        loss_norm = torch.mean(torch.norm(fore_weights, p=1, dim=2))

        loss_total = loss_base + loss_supp + cfg.TRAIN.C_LOSS_NORM * loss_norm

        loss_dict["loss_base"] = loss_base
        loss_dict["loss_supp"] = loss_supp
        loss_dict["loss_norm"] = loss_norm
        loss_dict["loss_total"] = loss_total

        return loss_total, loss_dict
