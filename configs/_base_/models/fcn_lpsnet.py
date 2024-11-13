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
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=96 * 2,  # channels[-1] * num_paths
        in_index=0,
        channels=96 * 2,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
        # Add the OHEMPixelSampler
        sampler=dict(
            type='OHEMPixelSampler',
            thresh=0.7,
            min_kept=10000,
        ),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))