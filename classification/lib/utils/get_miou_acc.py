import cv2
import numpy as np
import os
from PIL import Image
# import matplotlib.pyplot as plt

root = '/data/huangyx/data/HSICityV2/'

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = pred
    seg_gt = label

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index].astype('int32')
    seg_pred = seg_pred[ignore_index].astype('int32')

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix


def load(name):

    fine_label_path = os.path.join(root, 'test', 'rgb' + name + '_gray.png')
    cls_res_path = os.path.join('result/rgb_twocnn', name + '.png')

    flabel = cv2.imread(fine_label_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.imread(cls_res_path, cv2.IMREAD_GRAYSCALE)

    return flabel, res


def main():
    num_class = 19
    confusion_matrix = np.zeros(( num_class, num_class))

    lst_path = 'data/list/hsicity2/testval.lst'
    lst = [l.strip() for l in open(lst_path)]
    for i in lst:
        print(i[5:])
        name = i[5:]
        fine_label, cls_label0 = load(name)
        
        confusion_matrix += get_confusion_matrix(fine_label, cls_label0, size=fine_label.shape,
                                                 num_class=num_class, ignore=255)

    print(f'*******************')
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum() / pos.sum()
    acc_array = tp / np.maximum(1.0, pos)
    mean_acc = acc_array[1:].mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array[1:].mean()
    print(f'MIou: {IoU_array}')
    print(mean_IoU)
    print(f'acc: {acc_array}')
    print(mean_acc)


if __name__=='__main__':
    main()