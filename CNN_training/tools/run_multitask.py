#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# __author__ = 'junxi'

import main_debug
import yaml

# task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
iStart = 6 # define here.
iEnd = 6 # define here.


# FOR CONTACTED DATA
if __name__ == '__main__':
    # for rsn_i in range(2, 3):
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue
        with open('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/experiments/cls.yaml') as f:
            content = yaml.load(f)

            # output: <type 'dict'>
            # print(type(content))
            # print(content)

            content['NETWORK']['TOPK_K_R'] = 0.125  # default=0.125
            content['DATASET']['TRAIN_SPLIT'] = 'Contact_train_Group_for_major_review_1/%02d'%(rsn_i)
            content['DATASET']['VAL_SPLIT'] = 'Contact_train_Group_for_major_review_2/%02d' % (rsn_i)
            content['DATASET']['NUM_POINTS'] = 1940
            content['TRAIN']['OUTPUT_DIR'] = 'output/Contact_Group_for_major_review/%02d' % (rsn_i)
            content['TEST']['RESULT_DIR'] = 'output/Contact_Group_for_major_review/%02d' % (rsn_i)
            content['BASIC']['LOG_DIR'] = 'logs/Contact_Group_for_major_review/%02d'% (rsn_i)
            content['TRAIN']['EPOCH_NUM'] = 200

            # print(content)

        with open('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/experiments/cls.yaml', 'w') as nf:
            yaml.dump(content, nf)
        main_debug.main()
