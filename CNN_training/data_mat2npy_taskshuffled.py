import numpy as np
import scipy.io as sio
import os
import h5py
import random

task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
task_key = list(task_dict)
task_shuffle = random.sample(task_key,len(task_key))
ori_loc = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Train_MatData/Group_for_Contact_Train_adjusted/train_perc_0.5/'
iStart = 1 # define here.
iEnd = 10 # define here.

##### prepare group training task shuffled contacted data for CNN network ######
# task_shuffle = ['RELATIONAL', 'LANGUAGE', 'EMOTION', 'WM', 'SOCIAL', 'GAMBLING', 'MOTOR']
task_shuffle = ['GAMBLING', 'RELATIONAL', 'SOCIAL', 'MOTOR', 'EMOTION', 'WM', 'LANGUAGE']
print(task_shuffle)
for rsn_i in range(iStart, iEnd + 1):
    if rsn_i ==5:
        continue
    for task in task_shuffle:
        data = h5py.File('%s%s/RSN_%02d_train_perc_0.5.mat'%(ori_loc,task,rsn_i), 'r')
        features_task = np.array(data['fMRI_train']).T
        labels_task = np.array(data['label_train']).T
        # compute the vertices number
        # x = np.size(features, 0)
        if task == task_shuffle[0]:
            features = features_task
            labels = labels_task
        else:
            features = np.concatenate((features, features_task),axis=1)
            labels = np.concatenate((labels, labels_task),axis=1)

    for i in range(np.size(features, 0)):
        #i = 29197
        fpath = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/data/Contact_Data_taskshuffled_Group_1_for_test/%02d/'%(rsn_i)
        if (not os.path.exists(fpath)):
            os.makedirs(fpath)
        label = np.zeros(1)
        # label = np.zeros(1, dtype=np.int)
        label[0] = labels[i, 1]
        np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
    print('finished RSN_%02d task shuffled data transform!'%(rsn_i))
print('finished group1 task shuffled data transform!')

# task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
# task_key = list(task_dict)
# task_shuffle = random.sample(task_key,len(task_key))
# task_shuffle = ['GAMBLING', 'RELATIONAL', 'SOCIAL', 'MOTOR', 'EMOTION', 'WM', 'LANGUAGE']
task_shuffle = ['RELATIONAL', 'LANGUAGE', 'EMOTION', 'WM', 'SOCIAL', 'GAMBLING', 'MOTOR']
print(task_shuffle)
for rsn_i in range(iStart, iEnd + 1):
    if rsn_i ==5:
        continue
    for task in task_shuffle:
        data = h5py.File('%s%s/RSN_%02d_test_perc_0.5.mat'%(ori_loc,task,rsn_i), 'r')
        features_task = np.array(data['fMRI_test']).T
        labels_task = np.array(data['label_test']).T
        # compute the vertices number
        # x = np.size(features, 0)
        if task == task_shuffle[0]:
            features = features_task
            labels = labels_task
        else:
            features = np.concatenate((features, features_task),axis=1)
            labels = np.concatenate((labels, labels_task),axis=1)

    for i in range(np.size(features, 0)):
        fpath = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/data/Contact_Data_taskshuffled_Group_2_for_test/%02d/'%(rsn_i)
        if (not os.path.exists(fpath)):
            os.makedirs(fpath)
        label = np.zeros(1)
        # label = np.zeros(1, dtype=np.int)
        label[0] = labels[i, 1]
        np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
    print('finished RSN_%02d task shuffled data transform!'%(rsn_i))
print('finished group2 task shuffled data transform!')


