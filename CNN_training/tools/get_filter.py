import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import scipy.io as sio
import numpy as np

import _init_paths
from config.default import config as cfg
from models.network import Network

task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
iStart = 1 # define here.
iEnd = 10 # define here.
orig_path = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_2_EPOCH=500'

if __name__ == '__main__':
    # for rsn_i in range(2, 3):
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        model_fname = '%s/%02d/checkpoint_best.pth'%(orig_path, rsn_i)
        model = Network(cfg)
        # optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(model_fname)
        model_paramters = model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        a = model.conv_1.weight
        a = torch.squeeze(a)
        a = a.detach().numpy()
        sio.savemat('%s/%02d/conv_1.mat'%(orig_path, rsn_i), {'conv_1':a})
        b = model.conv_2.weight
        b = torch.squeeze(b)
        b = b.detach().numpy()
        sio.savemat('%s/%02d/conv_2.mat'%(orig_path, rsn_i), {'conv_2':b})
        c = model.pred.weight
        c = torch.squeeze(c)
        c = c.detach().numpy()
        sio.savemat('%s/%02d/conv_pred.mat'%(orig_path, rsn_i), {'conv_pred':c})

