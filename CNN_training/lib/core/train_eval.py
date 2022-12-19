import torch
import os
import numpy as np


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def train(cfg, data_loader, model, optimizer, criterion):
    model.train()

    loss_record = 0
    num_correct = 0
    num_total = 0
    for feature, label in data_loader:
        # feature: [128, 405] --> [128, 1, 405]
        feature = torch.unsqueeze(feature, dim=1)
        feature = feature.type_as(dtype)
        # label: [128, 1] --> [128]
        label = torch.squeeze(label)
        label_np = label.numpy()
        label = label.type_as(dtypel)

        prediction = model(feature)
        loss = criterion(prediction, label)
        loss_record += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())

        values, idxes = torch.max(prediction, dim=1)
        prediction_cls = idxes.data.cpu().numpy()
        correct_pred = np.sum(prediction_cls == label_np).astype(np.float32)
        num_correct += correct_pred
        num_total += label_np.shape[0]
    loss_average = loss_record / len(data_loader)
    train_acc = num_correct / num_total
    return loss_average, train_acc, prediction_cls


def evaluate(cfg, data_loader, model, epoch, criterion):
    model.eval()

    num_correct = 0
    num_total = 0
    loss_record = 0

    for feature, label in data_loader:
        # feature: [128, 405] --> [128, 1, 405]
        feature = torch.unsqueeze(feature, dim=1)
        feature = feature.type_as(dtype)
        # label: [128, 1] --> [128]
        label = torch.squeeze(label)
        label_np = label.numpy()
        label = label.type_as(dtypel)

        prediction = model(feature)

        loss = criterion(prediction, label)
        loss_record += loss.item()

        # Notice: we omit the softmax operation, as we only require the relative magnitude
        values, idxes = torch.max(prediction, dim=1)
        prediction_cls = idxes.data.cpu().numpy()
        correct_pred = np.sum(prediction_cls == label_np).astype(np.float32)
        num_correct += correct_pred
        num_total += label_np.shape[0]

    loss_average = loss_record / len(data_loader)
    test_acc = num_correct / num_total

    return loss_average, test_acc, prediction_cls
