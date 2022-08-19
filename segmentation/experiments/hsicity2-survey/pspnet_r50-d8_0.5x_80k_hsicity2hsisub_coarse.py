_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/hsicity2-hsi.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model=dict(
    backbone=dict(in_channels=128)
)

data_root = 'data/HSICityV2-subset/'

data=dict(
    train=dict(
        data_root=data_root,
        img_dir='train-coarse',
        ann_dir='train-coarse'),
    val=dict(
        data_root=data_root,
        img_dir='test',
        ann_dir='test'),
    test=dict(
        data_root=data_root,
        img_dir='test',
        ann_dir='test'),
)

runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
