import numpy as np
import scipy.io as sio
import os
iStart = 1 # define here.
iEnd = 10 # define here.

ori_loc = '/disk2/wqy/Projects/PyTorch_RSN/Train_MatData/Group_for_Contact_Train/train_perc_0.8/'
# for rsn_i in range(iStart, iEnd + 1):
rsn_i =2
fpath = '/disk2/wqy/Projects/PyTorch_RSN/CNN_training/data/Contact_Train_Data/%02d/'%(rsn_i)
data = sio.loadmat('%sEMOTION/RSN_%02d_train_perc_0.8.mat'%(ori_loc,rsn_i))
features = data['fMRI_train']
labels = data['label_train']

    # for i in range(np.size(features, 0)):
datas = np.load('%s29197.npz'%(fpath)) # 05648 15994 21462 18550
feature = datas['feature']
label = datas['label']
