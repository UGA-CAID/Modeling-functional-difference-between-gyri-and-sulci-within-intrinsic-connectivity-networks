import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from config.default import config as cfg
from config.default import update_config
import pprint
from models.network import Network
from dataset.dataset import ClsDataset
from core.train_eval import train, evaluate
from core.functions import prepare_env

from utils.utils import decay_lr, save_model, save_best_model, save_predictions_mat
from criterion.loss import BasNetLoss
from utils.utils import save_best_record_txt


def args_parser():
    parser = argparse.ArgumentParser(description='classification demo')
    parser.add_argument('-cfg', help='Experiment config file', default='../../experiments/orig_cls/cls_group2_1.yaml')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    update_config(args.cfg)
    if cfg.BASIC.SHOW_CFG:
        pprint.pprint(cfg)
    # prepare running environment for the whole project
    prepare_env(cfg)

    # log
    writer = SummaryWriter(log_dir=os.path.join(cfg.BASIC.ROOT_DIR, cfg.BASIC.LOG_DIR))

    # dataloader
    train_dset = ClsDataset(cfg, cfg.DATASET.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)
    val_dset = ClsDataset(cfg, cfg.DATASET.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.BASIC.WORKERS, pin_memory=cfg.BASIC.PIN_MEMORY)

    # network
    model = Network(cfg)
    model.cuda()

    # # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
    # todo: try Adam
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # # criterion
    # criterion = BasNetLoss()
    criterion = nn.CrossEntropyLoss()

    best_acc = -1

    for epoch in range(1, cfg.TRAIN.EPOCH_NUM+1):
#    for epoch in range(1,2):    #for test
        print('Epoch: %d:' % epoch)
        train_loss, train_acc, predictions = train(cfg, train_loader, model, optimizer, criterion)
        writer.add_scalar('train_loss/train', train_loss, epoch)
        print('Training loss %f' % train_loss)
        print('Training accuracy %f' % train_acc)
        # decay learning rate
        if epoch in cfg.TRAIN.LR_DECAY_EPOCHS:
            decay_lr(optimizer, factor=cfg.TRAIN.LR_DECAY_FACTOR)

        if epoch % cfg.TEST.EVAL_INTERVAL == 0:
            test_loss, test_acc, predictions = evaluate(cfg, val_loader, model, epoch, criterion) #predictions added
            print('test_loss %f' % test_loss)
            print('test_acc %f' % test_acc)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            # save_model(cfg, epoch=epoch, model=model, optimizer=optimizer)

            if test_acc > best_acc:
                best_acc = test_acc
                save_best_record_txt(epoch, train_acc, test_acc, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, "best_acc.txt"))
                save_best_model(cfg, epoch=epoch, model=model, optimizer=optimizer)
                #save_predictions_mat(predictions, os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.RESULT_DIR, "prediction.mat")) #predictions added

    writer.close()


if __name__ == '__main__':
    main()
