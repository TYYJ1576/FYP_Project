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

norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=(1536, 768),
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='LPSNet',
        in_channels=3,
        depths=[1, 3, 3, 10, 10],
        channels=[8, 24, 48, 96, 96],
        scale_ratios=[1.0, 0.25],
        conv_type='standard',
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, bias=0, layer=['BatchNorm2d', 'SyncBatchNorm'])
        ]
    ),
    decode_head=dict(
        type='ASPPHead',
        in_channels=96 * 2,  # channels[-1] * num_paths
        in_index=0,
        channels=256,  # Intermediate channels in the ASPPHead
        dilations=(1, 12, 24, 36),  # Atrous convolution dilation rates
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)