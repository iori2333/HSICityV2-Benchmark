import cv2
import os
import numpy as np
from lib.utils.utils import confidence_label_softmax


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    seg_pred = pred
    seg_gt = label

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

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


def erode_label(coarse_label, refined_label):
    mask = np.zeros(coarse_label.shape)
    kernel = [np.ones((9, 9), np.uint8),  # car
              np.ones((9, 9), np.uint8),  # human
              np.ones((9, 9), np.uint8),  # road
              np.ones((9, 9), np.uint8),  # light
              np.ones((9, 9), np.uint8),  # sign
              np.ones((9, 9), np.uint8),  # tree
              np.ones((9, 9), np.uint8),  # building
              np.ones((9, 9), np.uint8),  # sky
              np.ones((9, 9), np.uint8),  # object
              ]
    for i in range(1, 10):
        masklabel = np.where(coarse_label == i, 1, 0).astype(np.uint8)
        masklabel = cv2.erode(masklabel, kernel[i - 1])
        mask += masklabel
    label = coarse_label * mask + refined_label * (1 - mask)
    return label


coarse_label_list = [
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180056_110313_josn',
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180226_64001_josn',
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180241_58359_json',
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180535_7407_json',
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180641_79313_json',
    'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180919_20370_json',
]

refined_label_list = [
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180056_110313.png',
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180226_64001.png',
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180241_58359.png',
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180535_7407.png',
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180641_79313.png',
    'F:/spyder_code/HSIseg/result/test_val_results/confidence_softman0.5/20190528_180919_20370.png'
]
# refined_label_list = [
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180056_110313_josn/1.png',
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180226_64001_josn/1.png',
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180241_58359_json/1.png',
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180535_7407_json/1.png',
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180641_79313_json/1.png',
#     'F:/database/HSIcityscapes/testing_dataset/rgb20190528_180919_20370_json/1.png',
# ]


def fusionLabel(hsicla_label):
    num_class = 10
    coarse_label = []
    refined_label = hsicla_label
    ground_truth = []
    confusion_matrix = np.zeros((num_class, num_class))

    for i in coarse_label_list:
        coarse_label.append(np.array(cv2.imread(os.path.join(i, '2.png'),
                                                cv2.IMREAD_GRAYSCALE)))
        ground_truth.append(np.array(cv2.imread(os.path.join(i, 'label_gray.png'),
                                                cv2.IMREAD_GRAYSCALE)).transpose(1, 0)[:, ::-1])

    if not refined_label:
        refined_label = []
        for j in refined_label_list:
            refined_label.append(np.array(cv2.imread(j, cv2.IMREAD_GRAYSCALE)))

    fusion_lable = []
    for num in range(len(coarse_label)):
        fusion_lable.append(erode_label(coarse_label[num], refined_label[num]))

    for num in range(len(coarse_label)):
        confusion_matrix += get_confusion_matrix(ground_truth[num], fusion_lable[num], size=coarse_label[0].shape,
                                                 num_class=num_class, ignore=-1)
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


def load_data():
    val_hsicla_list = [
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180056_110313.npy',
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180226_64001.npy',
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180241_58359.npy',
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180535_7407.npy',
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180641_79313.npy',
        r'F:\spyder_code\HSIseg\result\test_val_results\20190528_180919_20370.npy'
    ]

    hsicla_result = []
    for i in val_hsicla_list:
        hsicla_result.append(np.load(i))
    return hsicla_result


if __name__ == '__main__':
    hsicla_data = load_data()
    # for thre in range(1, 10):
    thre = 0
    softmax_result = []
    for i in hsicla_data:
        softmax_result.append(confidence_label_softmax(i, threshold=thre/10))
    print(thre/10)
    fusionLabel(softmax_result)
