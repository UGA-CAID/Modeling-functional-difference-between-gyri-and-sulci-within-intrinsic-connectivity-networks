import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys


class ClsDataset(Dataset):
    def __init__(self, cfg, split):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.DATA_DIR)
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.base_dir = os.path.join(self.root, self.split)
        self.datas = self._make_dataset()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        file_name = self.datas[idx]
        data = np.load(os.path.join(self.base_dir, file_name))

        feature = data['feature']
        label = data['label']
        # print(file_name)
        return feature, label

    def _make_dataset(self):
        datas = os.listdir(self.base_dir)
        datas.sort() # sort the filename
        datas = [i for i in datas if i.endswith('.npz')]
        return datas


if __name__ == '__main__':

    sys.path.append('/disk2/wqy/Projects/Pytorch_RSN/lib')
    from config.default import config, update_config

    cfg_file = '/disk2/wqy/Projects/Pytorch_RSN/cls.yaml'
    update_config(cfg_file)
    train_dset = TALDataset(config, config.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True)

    for feature, label in train_loader:
        print(type(feature), feature.size(), type(label), label.size())
