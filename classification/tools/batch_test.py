import argparse
import os
import timeit
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from core.criterion import CrossEntropy
from core.function import testval_batched
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--path',
                        default='/home/huangyx/code/HSI2seg/data/')
                        # default='../data/')

    parser.add_argument('--output_dir',
                        default='output_batched_w21', type=str)
    parser.add_argument('--log_dir',
                        default='log_batched_w21', type=str)
    parser.add_argument('--model_file',
                        default='/home/huangyx/code/HSI2seg/output_rssan_w21/hsicity/hsicity2/final_state.pth', type=str)

    parser.add_argument('--model',
                        default='RSSAN')
    parser.add_argument('--model_name',
                        default='RSSAN')
    parser.add_argument('--resume',
                        default=False)
    parser.add_argument('--num_classes',
                        default=19, type=int)
    parser.add_argument('--ignore_label',
                        default=255, type=int)
    parser.add_argument('--test_row_size',
                        default=1, type=int)
    parser.add_argument('--window_size',
                        default=21, type=int)
    parser.add_argument('--exp_name',
                        default='hsicity2')

    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(
        args.output_dir, 'hsicity', 'batched', args.log_dir, args.exp_name, 'test'
    )

    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    device = torch.device('cuda:0')

    models = [
        models.RSSAN.RSSAN(3, 19, windows=args.windows_size),
        models.HybridSN.HybridSN(3, 19, windows=args.windows_size),
        models.JigSawHSI.JigSawHSI(3, 19, windows=args.windows_size),
    ]

    state_dicts = [
        '/home/huangyx/code/HSI2seg/output_rssan_w21/hsicity/hsicity2/final_state.pth',
        '/home/huangyx/code/HSI2seg/output_HybridSN_w21/hsicity/hsicity2/final_state.pth',
    ]

    for i in range(len(models)):
        model_state_file = state_dicts[i]
        pretrained_dict = torch.load(model_state_file)
        model_dict = models[i].state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()
                        if k[7:] in model_dict.keys()}

        model_dict.update(pretrained_dict)
        models[i].load_state_dict(model_dict)
        models[i] = models[i].to(device)

        logger.info('=> loading model {} from {}'.format(i, model_state_file))

    test_size = (1889, 1422)
    test_dataset = eval('datasets.hsicity2')(
        root='/data/huangyx/data/HSICityV2/',
        list_path='data/list/hsicity2/testval.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=args.ignore_label,
        base_size=1889,
        crop_size=test_size,
        center_crop_test=False,
        downsample_rate=1)
    
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    start = timeit.default_timer()
    ret = testval_batched(args.num_classes,
                args.ignore_label,
                args.test_row_size,
                test_dataset,
                testloader,
                models,
                args.window_size,
                sv_pred=False,
                sv_dir='result',
                )

    for i, r in enumerate(ret):
        mean_IoU, IoU_array, pixel_acc, mean_acc = r
        msg = 'model %d: MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
            Mean_Acc: {: 4.4f}, Class IoU: '.format(i, mean_IoU, pixel_acc, mean_acc)
        logging.info(msg)
        logging.info(IoU_array)

    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int((end - start) / 60))
    logger.info('Done')


if __name__ == '__main__':
    main()
