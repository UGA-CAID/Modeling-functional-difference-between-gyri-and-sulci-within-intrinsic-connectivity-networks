import torch
import numpy as np
import random
import os
import shutil
import scipy.io as sio


def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)


def backup_codes(root_dir, res_dir, backup_list):
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)
    os.makedirs(res_dir)
    for name in backup_list:
        shutil.copytree(os.path.join(root_dir, name), os.path.join(res_dir, name))
    print('codes backup at {}'.format(os.path.join(res_dir, name)))


def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def save_model(cfg, epoch, model, optimizer):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    save_name = 'checkpoint_'+str(epoch).zfill(cfg.BASIC.CHECKPOINT_DIGITS)+'.pth'
    save_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR, save_name)
    torch.save(state, save_file)
    print('save model: %s' % save_file)
    return save_file


def save_best_model(cfg, epoch, model, optimizer):
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    # save_name = 'checkpoint_'+str(epoch).zfill(cfg.BASIC.CHECKPOINT_DIGITS)+'.pth'
    save_name = 'checkpoint_'+'best'+'.pth'
    save_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TRAIN.OUTPUT_DIR, save_name)
    torch.save(state, save_file)
    print('save model: %s' % save_file)
    return save_file


def load_weights(model, weight_file):
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def save_best_record_txt(epoch, train_acc, test_acc, file_path):
    fo = open(file_path, "w")
    fo.write("Epoch: {}\n".format(epoch))
    fo.write("Train_acc: {:.5f}\n".format(train_acc))
    fo.write("Test_acc: {:.5f}\n".format(test_acc))
    fo.close()

def save_predictions_mat(predictions, file_path):
    foo = {'predicted': predictions}
    sio.savemat(file_path, foo)
