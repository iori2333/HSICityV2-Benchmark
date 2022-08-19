_base_ = [
    '../_base_/datasets/hsicity2.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FuseNet',
        in_channels=(3, 128),
    ),
    decode_head=dict(
        type='FuseNetHead',
        num_classes=19,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=1, workers_per_gpu=1)