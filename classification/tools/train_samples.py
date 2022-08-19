import argparse
import os
import timeit
import logging
import shutil
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
from core.function import train, validate
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--path',
                        default='data')
    parser.add_argument('--output_dir',
                        default='output_rgb_cnnhsi', type=str)
    parser.add_argument('--log_dir',
                        default='log_rgb_cnnhsi', type=str)

    parser.add_argument('--model',
                        default='CNN_HSI')
    parser.add_argument('--model_name',
                        default='CNN_HSI')
    parser.add_argument('--resume',
                        default=True)
    parser.add_argument('--num_classes',
                        default=19, type=int)
    parser.add_argument('--ignore_label',
                        default=255, type=int)
    parser.add_argument('--learning_rate',
                        default=0.001, type=float)
    parser.add_argument('--batch_size',
                        default=8, type=int)
    parser.add_argument('--window_size',
                        default=5, type=int)
    parser.add_argument('--epoch',
                        default=16, type=int)
    parser.add_argument('--print_freq',
                        default=20, type=int)

    parser.add_argument('--exp_name',
                        default='hsicity2')
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger, final_output_dir, log_dir = create_logger(
        args.output_dir, 'hsicity', args.model, args.log_dir, args.exp_name, 'train'
    )
    print('## running on', args.local_rank)

    writer_dict = {
        'writer': SummaryWriter(args.log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    gpus = (0, 1)
    distributed = len(gpus) > 1
    device = torch.device(f'cuda:{args.local_rank}')

    # build model
    model = eval('models.' + args.model + '.' +
                   args.model_name)(128, 19, windows=args.window_size)
    if args.local_rank == 0:
        # copy model file
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # prepare data
    crop_size = (1889, 1422)

    train_dataset = eval('datasets.hsicity2')(
        root='/data/huangyx/data/HSICityV2/',
        list_path='data/list/hsicity2/train.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=True,
        ignore_label=args.ignore_label,
        base_size=1889,
        crop_size=crop_size,
        scale_factor=10
    )

    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    test_size = (1889, 1422)
    test_dataset = eval('datasets.hsicity2')(
        root='/data/huangyx/data/HSICityV2/',
        list_path='data/list/hsicity2/val_temp.lst',
        num_samples=None,
        num_classes=args.num_classes,
        multi_scale=False,
        flip=False,
        ignore_label=args.ignore_label,
        base_size=1889,
        crop_size=test_size,
        center_crop_test=False,
        downsample_rate=1)

    if distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        sampler=test_sampler)

    criterion = nn.CrossEntropyLoss(weight=train_dataset.class_weights,
                                    ignore_index=args.ignore_label)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.SGD([{'params':
                                filter(lambda p: p.requires_grad,
                                 model.parameters()),
                                'lr': args.learning_rate}],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0005,
                                nesterov=False)
 
    best_mIoU = 0
    last_epoch = 0
    if args.resume:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file,
                                    map_location=lambda storage, loc: storage)
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))

    epoch_iters = int(len(train_dataset) / args.batch_size / len(gpus))
    start = timeit.default_timer()
    end_epoch = args.epoch
    window_size = args.window_size

    for epoch in range(last_epoch, end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        train(epoch, end_epoch, args.print_freq,
              epoch_iters, args.learning_rate,
              trainloader, optimizer, criterion, model, writer_dict, device, window_size)

        valid_loss, mean_IoU, IoU_array = validate(
            args.num_classes, args.ignore_label,
            testloader, criterion,
            model, writer_dict,
            device, window_size)

        if args.local_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))

            torch.save(model.state_dict(),
                           os.path.join(final_output_dir, 'last_state.pth'))

            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

            if epoch == end_epoch - 1:
                torch.save(model.state_dict(),
                           os.path.join(final_output_dir, 'final_state.pth'))
                writer_dict['writer'].close()
                end = timeit.default_timer()
                logger.info('Hours: %d' % np.int((end - start) / 3600))
                logger.info('Done')


if __name__ == '__main__':
    main()
