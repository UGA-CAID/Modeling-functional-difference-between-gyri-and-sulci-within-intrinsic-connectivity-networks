# ------------------------------------------------------------------------------
# Author: Le Yang
# Descriptions: This is a simplified version of main.py, mainly used for debug, especially the evaluation process
# ------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import _init_paths
from config.default import config as cfg
from config.default import update_config
from models.network import Network
from dataset.dataset import ClsDataset
from core.train_eval import train, evaluate
#from core.functions import prepare_env, evaluate_mAP


def args_parser():
    parser = argparse.ArgumentParser(description='classification demo')
    parser.add_argument('-cfg', help='Experiment config file', default='../experiments/2corssval_test_on_sub_contacted.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)

    val_dset = ClsDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = Network(cfg)
    model.cuda()

    weight_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.MODEL_DIR, 'checkpoint_best.pth')
    epoch = 100
    criterion = nn.CrossEntropyLoss()
    from utils.utils import load_weights, save_predictions_mat
    model = load_weights(model, weight_file)
    test_loss, test_acc, predictions = evaluate(cfg, val_loader, model, epoch, criterion)
    save_predictions_mat(predictions,
                         os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, "prediction.mat"))  # predictions added

    #evaluate_mAP(cfg, actions_json_file, os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE))


if __name__ == '__main__':
    main()

