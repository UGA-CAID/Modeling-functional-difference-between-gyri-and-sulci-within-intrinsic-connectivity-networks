#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import main_eval
import yaml
import os
import numpy as np
import scipy.io as sio

task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
# task_dict = {'RELATIONAL': 232, 'LANGUAGE': 316, 'EMOTION': 176, 'WM': 405,  'SOCIAL': 274, 'GAMBLING': 253, 'MOTOR': 284}
# task_dict = {'GAMBLING': 253, 'RELATIONAL': 232, 'SOCIAL': 274, 'MOTOR': 284, 'EMOTION': 176, 'WM': 405, 'LANGUAGE': 316, }
# sub_list = [1:16 18 20:33 35:43 45:51 53:60 62:68]
ori_loc = '/data/hzb1/Projects/S900_RSN/Data/for_analysis/'
iStart = 1 # define here.
iEnd = 10 # define here.

# prepare sub list
fname = '/data/hzb1/Projects/S900_RSN/sub_list.mat'
sub_list = sio.loadmat(fname)
sub_list = sub_list['sub_list']
# print(sub_list)

# contacted data test
if __name__ == '__main__':
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        for sub_i in sub_list:
            for task in task_dict:
                fname = '%sRSN_%02d/%s/%d.RSN_GS.mat' % (ori_loc, rsn_i, task, sub_i)
                data = sio.loadmat(fname)
                features_task = data['X'].T
                labels_task = data['Y']
                # if task == 'EMOTION':
                # if task == 'RELATIONAL':
                if task == 'GAMBLING':
                    features = features_task
                    labels = labels_task
                else:
                    features = np.concatenate((features, features_task),axis=1)
                    labels = np.concatenate((labels, labels_task),axis=1)

            for i in range(np.size(features, 0)):
                # i = 29197
                fpath = '/data/hzb1/Projects/S900_RSN/CNN_training/data/test_on_sub_contacted/%02d/%d/' % (rsn_i, sub_i)
                if (not os.path.exists(fpath)):
                    os.makedirs(fpath)
                    continue
                label = np.zeros(1)
                # label = np.zeros(1, dtype=np.int)
                label[0] = labels[i, 1]
                np.savez('%s%05d.npz' % (fpath, i + 1), feature=features[i, :], label=label)

            with open('/data/hzb1/Projects/S900_RSN/CNN_training/experiments/2corssval_test_on_sub_contacted.yaml') as f:
                content = yaml.load(f)

                content['DATASET']['VAL_SPLIT'] = 'test_on_sub_contacted/%02d/%d/' % (rsn_i, sub_i)
                content['DATASET']['NUM_POINTS'] = 1940
                content['TEST']['BATCH_SIZE'] = np.size(features, 0)
                content['TEST']['MODEL_DIR'] = 'output/Contact_Data_taskshuffled_group_2/%02d/' % (rsn_i)
                content['TEST']['RESULT_DIR'] = 'output/Contact_Data_taskshuffled_group_2/test_results/%02d/%d/' % (rsn_i, sub_i)
                pre_path = '/data/hzb1/Projects/S900_RSN/CNN_training/output/Contact_Data_taskshuffled_group_2/test_results/%02d/%d/' % (rsn_i, sub_i)
                if (not os.path.exists(pre_path)):
                    os.makedirs(pre_path)
                # print(content)

            with open('/data/hzb1/Projects/S900_RSN/CNN_training/experiments/2corssval_test_on_sub_contacted.yaml', 'w') as nf:
                yaml.dump(content, nf)
            main_eval.main()