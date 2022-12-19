#!/usr/bin/env python
# _*_ coding:utf-8 _*_

import main_eval_2
import yaml
import os
import numpy as np
import scipy.io as sio

task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
# sub_list = [1:16 18 20:33 35:43 45:51 53:60 62:68]
ori_loc = '/data1/wqy/Projects/S900_RSN/Data/for_analysis/'
iStart = 1 # define here.
iEnd = 10 # define here.

# prepare sub list
fname = '/data1/wqy/Projects/S900_RSN/sub_list.mat'
sub_list = sio.loadmat(fname)
sub_list = sub_list['sub_list']
# print(sub_list)

# contacted data test
if __name__ == '__main__':
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        for sub_i in sub_list:

            path = '/data1/wqy/Projects/S900_RSN/CNN_training/data/test_on_sub_contacted/%02d/%d/'% (rsn_i, sub_i)
            vertices_num = 0
            for fn in os.listdir(path):
                vertices_num += 1


            with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/2corssval_test_on_sub_contacted_2.yaml') as f:
                content = yaml.load(f)

                content['DATASET']['VAL_SPLIT'] = 'test_on_sub_contacted/%02d/%d/' % (rsn_i, sub_i)
                content['DATASET']['NUM_POINTS'] = 1940
                content['TEST']['BATCH_SIZE'] = vertices_num
                content['TEST']['MODEL_DIR'] = 'output/2-fold_CrossVal_group2/%02d/' % (rsn_i)
                content['TEST']['RESULT_DIR'] = 'output/2-fold_CrossVal_group2/test_results/%02d/%d/' % (rsn_i, sub_i)
                pre_path = '/data1/wqy/Projects/S900_RSN/CNN_training/output/2-fold_CrossVal_group2/test_results/%02d/%d/' % (rsn_i, sub_i)
                if (not os.path.exists(pre_path)):
                    os.makedirs(pre_path)
                # print(content)

            with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/2corssval_test_on_sub_contacted_2.yaml', 'w') as nf:
                yaml.dump(content, nf)
            main_eval_2.main()