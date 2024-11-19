_base_ = [
    '../../_base_/models/fcn_lpsnet.py',
    '../../_base_/datasets/cityscapes.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_90k.py',
]

crop_size = (1536, 768)

train_pipeline = [
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        resize_type='ResizeStepScaling',
        step_size=0.25,
        keep_ratio=True,
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=0.4,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18,
    )
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2
    )