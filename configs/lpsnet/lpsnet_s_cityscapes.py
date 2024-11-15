_base_ = [
    '../_base_/models/fcn_lpsnet.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_90k.py',
]

model = dict(
    backbone=dict(
        type='LPSNet',
        in_channels=3,
        depths=[1, 3, 3, 10, 10],
        channels=[8, 24, 48, 96, 96],
        scale_ratios=[0.75, 0.25],
    ),
)

# Data pipeline settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1536, 768),
        ratio_range=(0.5, 2.0),
        resize_type='ResizeStepScaling',
        step_size=0.25,
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=(1536, 768), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=0.4,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18,
    )
]
