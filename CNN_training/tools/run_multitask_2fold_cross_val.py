#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# __author__ = 'junxi'

import main_debug
import yaml

# task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
iStart = 1 # define here.
iEnd = 10 # define here.

if __name__ == '__main__':
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/cls_2crossval.yaml') as f:
            content = yaml.load(f)

            # output: <type 'dict'>
            # print(type(content))
            # print(content)

            content['DATASET']['TRAIN_SPLIT'] = 'Group_for_2-fold_CrossVal_train/%02d'%(rsn_i)
            content['DATASET']['VAL_SPLIT'] = 'Group_for_2-fold_CrossVal_test/%02d' % (rsn_i)
            content['DATASET']['NUM_POINTS'] = 1940
            content['TRAIN']['OUTPUT_DIR'] = 'output/2-fold_CrossVal_group1/%02d' % (rsn_i)
            content['TEST']['RESULT_DIR'] = 'output/2-fold_CrossVal_group1/%02d' % (rsn_i)
            content['BASIC']['LOG_DIR'] = 'logs/2-fold_CrossVal_group1/%02d'% (rsn_i)

            # print(content)

        with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/cls_2crossval.yaml', 'w') as nf:
            yaml.dump(content, nf)
        main_debug.main()

#exchange the group and train the network
    print('group 2 training begain!')
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/cls_2crossval.yaml') as f:
            content = yaml.load(f)

            content['DATASET']['TRAIN_SPLIT'] = 'Group_for_2-fold_CrossVal_test/%02d'%(rsn_i)
            content['DATASET']['VAL_SPLIT'] = 'Group_for_2-fold_CrossVal_train/%02d' % (rsn_i)
            content['DATASET']['NUM_POINTS'] = 1940
            content['TRAIN']['OUTPUT_DIR'] = 'output/2-fold_CrossVal_group2/%02d' % (rsn_i)
            content['TEST']['RESULT_DIR'] = 'output/2-fold_CrossVal_group2/%02d' % (rsn_i)
            content['BASIC']['LOG_DIR'] = 'logs/2-fold_CrossVal_group2/%02d'% (rsn_i)

        with open('/data1/wqy/Projects/S900_RSN/CNN_training/experiments/cls_2crossval.yaml', 'w') as nf:
            yaml.dump(content, nf)
        main_debug.main()