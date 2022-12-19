import numpy as np
import scipy.io as sio
import os
import csv
# import pandas as pd


task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
iStart = 1 # define here.
iEnd = 10 # define here.

# ori_loc = '/data/hzb1/Projects/Marmoset/FCN/output/cross_val_auditory2visual_minnum_RemoveTP_ION/'
ori_loc = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1_k=0.9/'
# fcsv = open('/data/hzb1/Projects/Marmoset/FCN/output/individual/best_acc', 'w')

with open('/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/output/Contact_Group_1_k=0.9/final_acc.csv', 'w') as fcsv:
    writer = csv.writer(fcsv)
    for rsn_i in range(iStart, iEnd + 1):
        if rsn_i == 5:
            continue

        f = open('%s%02d/best_acc.txt' %(ori_loc, rsn_i))
        ff = f.read()
        ff_list = ff.split()
        ff_list.append(rsn_i)
        writer.writerow(ff_list)
