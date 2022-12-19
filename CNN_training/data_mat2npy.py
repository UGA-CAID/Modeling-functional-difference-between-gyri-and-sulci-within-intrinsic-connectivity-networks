import numpy as np
import scipy.io as sio
import os
import h5py
import random

task_dict = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
task_key = list(task_dict)
task_shuffle = random.sample(task_key,len(task_key))
ori_loc = '/disk2/wqy/Projects/PyTorch_RSN/Train_MatData/Group_for_Orig_Train/train_perc_0.8/'
iStart = 6 # define here.
iEnd = 7 # define here.

# ###### prepare group training data for CNN network ######
# for task in task_dict:
#     for rsn_i in range(iStart, iEnd + 1):
#         data =  sio.loadmat('%s%s/RSN_%02d_train_perc_0.8.mat'%(ori_loc,task,rsn_i))
#         features = data['fMRI_train']
#         labels = data['label_train']
#         # compute the vertices number
#         # x = np.size(features, 0)
#
#         for i in range(np.size(features, 0)):
#             fpath = '/disk2/wqy/Projects/PyTorch_RSN/CNN_training/data/adjusted_train/%s/%02d/'%(task, rsn_i)
#             if (not os.path.exists(fpath)):
#                 os.makedirs(fpath)
#             label=labels[i, :]
#             np.savez('%s%05d.npz'%(fpath, i+1), feature=features[i, :], label=labels[i, :])
#         print('finished %s RSN_%02d trainset data transform!'%(task, rsn_i))
#
# for task in task_dict:
#     for rsn_i in range(iStart, iEnd + 1):
#         data =  sio.loadmat('%s%s/RSN_%02d_test_perc_0.2.mat'%(ori_loc,task,rsn_i))
#         features = data['fMRI_test']
#         labels = data['label_test']
#         # compute the vertices number
#         # x = np.size(features, 0)
#
#         for i in range(np.size(features, 0)):
#             fpath = '/disk2/wqy/Projects/PyTorch_RSN/CNN_training/data/adjusted_test/%s/%02d/'%(task, rsn_i)
#             if (not os.path.exists(fpath)):
#                 os.makedirs(fpath)
#             np.savez('%s%05d.npz'%(fpath, i+1), feature=features[i, :], label=labels[i, :])
#         print('finished %s RSN_%02d testset data transform!'%(task, rsn_i))

##### prepare test on subjects data for CNN network ######
# ori_loc = '/disk1/wqy/Projects/CNN_HCP_RSN/Data/for_analysis/'
# for task in task_dict:
#     for rsn_i in range(iStart, iEnd + 1):
#         for sub_i in range(1,69):
#             fname = '%sRSN_%02d/%s/%d.RSN_GS.mat'%(ori_loc, rsn_i, task, sub_i)
#             if (not os.path.isfile(fname)):
#                 print('skip')
#                 continue
#             data =  sio.loadmat(fname)
#
#             features = data['X'].T
#             labels = data['Y']
#             # compute the vertices number
#             # x = np.size(features, 0)
#
#             for i in range(np.size(features, 0)):
#                 fpath = '/disk2/wqy/Projects/PyTorch_RSN/CNN_training/data/test_on_sub/%s/%02d/%d/'%(task, rsn_i,sub_i)
#                 if (not os.path.exists(fpath)):
#                     os.makedirs(fpath)
#                 np.savez('%s%05d.npz'%(fpath, i+1), feature=features[i, :], label=labels[i, :])
#             print('finished %s RSN_%02d subject %d data transform!'%(task, rsn_i, sub_i))

#### prepare group training contacted data for CNN network ######
# ori_loc = '/data/wqy/Projects/S900_RSN/Train_MatData/Group_for_Contact_Train/train_perc_0.8/'
# for rsn_i in range(iStart, iEnd + 1):
#     if rsn_i ==5:
#         continue
#     #rsn_i = 2
#     # features = np.empty()
#     # labels = np.empty()
#     for task in task_dict:
#         #data =  sio.loadmat('%s%s/RSN_%02d_train_perc_0.8.mat'%(ori_loc,task,rsn_i))
#         data =  h5py.File('%s%s/RSN_%02d_train_perc_0.8.mat'%(ori_loc,task,rsn_i), 'r')
#         features_task = np.array(data['fMRI_train']).T
#         labels_task = np.array(data['label_train']).T
#         # compute the vertices number
#         # x = np.size(features, 0)
#         if task == 'EMOTION':
#             features = features_task
#             labels = labels_task
#         else:
#             features = np.concatenate((features, features_task),axis=1)
#             labels = np.concatenate((labels, labels_task),axis=1)
#
#     for i in range(np.size(features, 0)):
#         #i = 29197
#         fpath = '/data1/wqy/Projects/S900_RSN/CNN_training/data/Contact_Train_Data/%02d/'%(rsn_i)
#         if (not os.path.exists(fpath)):
#             os.makedirs(fpath)
#         label = np.zeros(1)
#         # label = np.zeros(1, dtype=np.int)
#         label[0] = labels[i, 1]
#         np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
#     print('finished RSN_%02d trainset data transform!'%(rsn_i))
#
# for rsn_i in range(iStart, iEnd + 1):
#     if rsn_i ==5:
#         continue
#     for task in task_dict:
#         data = h5py.File('%s%s/RSN_%02d_test_perc_0.2.mat'%(ori_loc,task,rsn_i), 'r')
#         features_task = np.array(data['fMRI_test'])
#         labels_task = np.array(data['label_test'])
#         # compute the vertices number
#         # x = np.size(features, 0)
#         if task == 'EMOTION':
#             features = features_task
#             labels = labels_task
#         else:
#             features = np.concatenate((features, features_task),axis=1)
#             labels = np.concatenate((labels, labels_task),axis=1)
#
#     for i in range(np.size(features, 0)):
#         fpath = '/data1/wqy/Projects/S900_RSN/CNN_training/data/Contact_Test_Data/%02d/'%(rsn_i)
#         if (not os.path.exists(fpath)):
#             os.makedirs(fpath)
#         label = np.zeros(1)
#         # label = np.zeros(1, dtype=np.int)
#         label[0] = labels[i, 1]
#         np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
#     print('finished RSN_%02d testset data transform!'%(rsn_i))
# print('finished task contacted data prepare and task shuffled contacted data prepartion begain!')
#
# ###### prepare group training task shuffled contacted data for CNN network ######
# ori_loc = '/data/wqy/Projects/S900_RSN/Train_MatData/Group_for_Contact_Train/train_perc_0.8/'
# for rsn_i in range(iStart, iEnd + 1):
#     if rsn_i ==5:
#         continue
#     for task in task_shuffle:
#         data =  h5py.File('%s%s/RSN_%02d_train_perc_0.8.mat'%(ori_loc,task,rsn_i), 'r')
#         features_task = np.array(data['fMRI_train']).T
#         labels_task = np.array(data['label_train']).T
#         # compute the vertices number
#         # x = np.size(features, 0)
#         if task == task_shuffle[0]:
#             features = features_task
#             labels = labels_task
#         else:
#             features = np.concatenate((features, features_task),axis=1)
#             labels = np.concatenate((labels, labels_task),axis=1)
#
#     for i in range(np.size(features, 0)):
#         #i = 29197
#         fpath = '/data1/wqy/Projects/S900_RSN/CNN_training/data/Contact_Train_Data_taskshuffled/%02d/'%(rsn_i)
#         if (not os.path.exists(fpath)):
#             os.makedirs(fpath)
#         label = np.zeros(1)
#         # label = np.zeros(1, dtype=np.int)
#         label[0] = labels[i, 1]
#         np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
#     print('finished RSN_%02d trainset data transform!'%(rsn_i))
#
# for rsn_i in range(iStart, iEnd + 1):
#     if rsn_i ==5:
#         continue
#     for task in task_shuffle:
#         data = h5py.File('%s%s/RSN_%02d_test_perc_0.2.mat'%(ori_loc,task,rsn_i), 'r')
#         features_task = np.array(data['fMRI_test'])
#         labels_task = np.array(data['label_test'])
#         # compute the vertices number
#         # x = np.size(features, 0)
#         if task == task_shuffle[0]:
#             features = features_task
#             labels = labels_task
#         else:
#             features = np.concatenate((features, features_task),axis=1)
#             labels = np.concatenate((labels, labels_task),axis=1)
#
#     for i in range(np.size(features, 0)):
#         fpath = '/data1/wqy/Projects/S900_RSN/CNN_training/data/Contact_Test_Data_taskshuffled/%02d/'%(rsn_i)
#         if (not os.path.exists(fpath)):
#             os.makedirs(fpath)
#         label = np.zeros(1)
#         # label = np.zeros(1, dtype=np.int)
#         label[0] = labels[i, 1]
#         np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
#     print('finished RSN_%02d testset data transform!'%(rsn_i))
# print('finished task shuffled contacted data and 200 sub NNadjusted data prepartion begain!')

###### prepare group training 200subj network adjusted contacted data for CNN network ######
ori_loc = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/Train_MatData/Group_for_Contact_Train_for_major_review/train_perc_0.5/'
for rsn_i in range(iStart, iEnd + 1):
    if rsn_i ==5:
        continue
    for task in task_dict:
        data = h5py.File('%s%s/RSN_%02d_train_perc_0.5.mat'%(ori_loc,task,rsn_i), 'r')
        features_task = np.array(data['fMRI_train']).T
        labels_task = np.array(data['label_train']).T
        # compute the vertices number
        # x = np.size(features, 0)
        if task == 'EMOTION':
            features = features_task
            labels = labels_task
        else:
            features = np.concatenate((features, features_task),axis=1)
            labels = np.concatenate((labels, labels_task),axis=1)

    for i in range(np.size(features, 0)):
        #i = 29197
        fpath = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/data/Contact_train_Group_for_major_review_1/%02d/'%(rsn_i)
        if (not os.path.exists(fpath)):
            os.makedirs(fpath)
        label = np.zeros(1)
        # label = np.zeros(1, dtype=np.int)
        label[0] = labels[i, 1]
        np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
    print('finished RSN_%02d trainset data transform!'%(rsn_i))

for rsn_i in range(iStart, iEnd + 1):
    if rsn_i ==5:
        continue
    for task in task_dict:
        data = h5py.File('%s%s/RSN_%02d_test_perc_0.5.mat'%(ori_loc,task,rsn_i), 'r')
        features_task = np.array(data['fMRI_test']).T
        labels_task = np.array(data['label_test']).T
        # compute the vertices number
        # x = np.size(features, 0)
        if task == 'EMOTION':
            features = features_task
            labels = labels_task
        else:
            features = np.concatenate((features, features_task),axis=1)
            labels = np.concatenate((labels, labels_task),axis=1)

    for i in range(np.size(features, 0)):
        fpath = '/data/hzb1/Projects/S900_RSN/New_Adjusted_GSextract_order/CNN_training/data/Contact_train_Group_for_major_review_2/%02d/'%(rsn_i)
        if (not os.path.exists(fpath)):
            os.makedirs(fpath)
        label = np.zeros(1)
        # label = np.zeros(1, dtype=np.int)
        label[0] = labels[i, 1]
        np.savez('%s%06d.npz'%(fpath, i+1), feature=features[i, :], label=label)
    print('finished RSN_%02d testset data transform!'%(rsn_i))