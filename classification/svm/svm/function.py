from copyreg import pickle
import logging
import os
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.svm import SVC
from dataset import HSICity2
from torch.utils.data import DataLoader
import gc
from utils.confusion_matrix import get_confusion_matrix
from tqdm import tqdm
import cv2

def get_average(data: np.ndarray, label: np.ndarray, sz: int = 1):
    b, c, h, w = data.shape
    cubes = data.squeeze(0).permute(1, 2, 0)
    label = label.squeeze(0)

    cubes = cubes.reshape((-1, c))
    label = label.reshape(-1)
    cubes = cubes[label != 255]
    label = label[label != 255]
    stat = [None for i in range(19)]
    for i in range(19):
        stat[i] = cubes[label == i].mean(dim=0)

    return stat


def train_average(model: SVC, train_loader: DataLoader, sz: int = 11):
    X_train = []
    y_train = []

    try:
        logging.info('try loading dataset')
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
    except IOError:
        logging.info('dataset not found, generating')
        n = len(train_loader)
        X_train, y_train = [], []
        for i, (image, label, hsi) in enumerate(train_loader):
            logging.info(f'getting average {i}/{n}')
            cubes = get_average(hsi, label, sz)
            for i, cube in enumerate(cubes):
                if not torch.isnan(cube).any():
                    X_train.append(cube.numpy())
                    y_train.append(i)
        logging.info(f'training {len(X_train)} cubes')
        gc.collect()
        X_train, y_train = np.array(X_train), np.array(y_train)
        np.save('X_train.npy', X_train)
        np.save('y_train.npy', y_train)
    model.fit(X_train, y_train)
    logging.info('fit done!')
    logging.info(f'train_acc = {model.score(X_train, y_train)}')
    del X_train, y_train
    gc.collect()
    return model


def async_val(model: SVC, i: int, label: np.ndarray, image: torch.Tensor, sz: int, out: str):
    b, c, h, w = image.shape
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    label = label.squeeze(0).numpy()

    ret = cv2.imread(f'{out}/ret-{i}.png', cv2.IMREAD_GRAYSCALE)
    if ret is None:
        cubes = image.reshape((-1, c))
        print(cubes.shape)
        pred = model.predict(cubes)
        ret = pred.reshape((h, w))

        cv2.imwrite(f'{out}/ret-{i}.png', ret)
    # for i in tqdm(range(sz, h, sz)):
    #     for j in range(sz, w, sz):
    #         cube_hsi = hsi[:, i - sz : i, j - sz : j]
    #         cube_hsi = cube_hsi.reshape((-1,))
    #         pred = model.predict(cube_hsi.reshape((1, -1)))
    #         ret[i - sz : i, j - sz : j] = pred
    current = get_confusion_matrix(label, ret, size=(h, w), num_class=19)
    pos = current.sum(0)
    res = current.sum(1)
    tp = np.diag(current)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = tp / np.maximum(1.0, pos + res - tp)
    mean_IoU = IoU_array.mean()
    logging.info(IoU_array)
    logging.info(f"mean_IoU = {mean_IoU}")
    logging.info(f"mean_acc = {mean_acc}")
    logging.info(f"pixel_acc = {pixel_acc}")

    return current

def validate(model: SVC, test_loader: DataLoader, sz: int, out: str):
    confusion_matrix = np.zeros((19, 19))
    n = len(test_loader)
    for i, (image, label, hsi) in enumerate(test_loader):
        logging.info(f'validating image {i}/{n}')
        current = async_val(model, i, label, hsi, sz, out)
        confusion_matrix += current

    logging.info('validation done!')
    pos = confusion_matrix.sum(0)
    res = confusion_matrix.sum(1)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    mean_acc = (tp / np.maximum(1.0, pos)).mean()
    IoU_array = tp / np.maximum(1.0, pos + res - tp)
    mean_IoU = IoU_array.mean()
    logging.info(IoU_array)
    logging.info(f"mean_IoU = {mean_IoU}")
    logging.info(f"mean_acc = {mean_acc}")
    logging.info(f"pixel_acc = {pixel_acc}")

    # print(model.score(X_test, y_test))
    return model

if __name__ == '__main__':
    data = torch.zeros((1, 128, 512, 512))
    label = torch.randint(0, 20, (512, 512))
    label[label == 19] = 255
    get_average(data, label)
    