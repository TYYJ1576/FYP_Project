_base_ = [
    '../_base_/models/lpsnet.py',
    '../_base_/datasets/cityscapes_768x768.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_400.py'
]

model = dict(
    backbone = dict(depth = (1, 3, 3, 10, 10), width = (8, 24, 48, 96, 96), resolution = (3 / 4, 1 / 4, 0)),
    decode_head = dict(in_channels = 192)
)