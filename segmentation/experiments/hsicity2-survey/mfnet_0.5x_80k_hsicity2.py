_base_ = [
    '../_base_/datasets/hsicity2.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

rgb_ch = (64, 64, 64, 256, 256)
inf_ch = (128, 128, 128, 256, 256)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MFNet',
        in_channels=(3, 128),
        rgb_ch=rgb_ch,
        inf_ch=inf_ch,
    ),
    decode_head=dict(
        type='MFNetHead',
        num_classes=19,
        in_index=[0, 1, 2, 3],
        rgb_ch=rgb_ch,
        inf_ch=inf_ch,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=1, workers_per_gpu=1)