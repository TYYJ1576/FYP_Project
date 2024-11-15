from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

# Create a new function based on F.interpolate but with some parameters fixed
# F.interpolate: Resizes or upscales/downscales a tensor
# mode="billinear": 
_interpolate = partial(F.interpolate, mode="bilinear", align_corners=True)


def _multipath_interaction(feats):
    length = len(feats)
    if length == 1:
        return feats
    sizes = [x.shape[-2:] for x in feats]
    outs = []
    indices = list(range(length))
    for i, s in enumerate(sizes):
        out = feats[i]
        for j in filter(lambda x: x != i, indices):
            out = out + _interpolate(feats[j], size=s)
        outs.append(out)
    return outs


@MODELS.register_module()
class LPSNet(BaseModule):

    def __init__(
        self,
        depths,
        channels,
        scale_ratios,
        in_channels=3,
        init_cfg=[
            dict(type='Kaiming', layer='Conv2d'),
            dict(type='Constant', val=1, layer=['BatchNorm2d', 'SyncBatchNorm'])
        ],
    ):
        super().__init__(init_cfg)

        self.depths = depths
        self.channels = channels
        self.scale_ratios = [r for r in scale_ratios if r > 0]
        self.in_channels = in_channels

        self.num_paths = len(self.scale_ratios)
        self.num_blocks = len(depths)

        if self.num_blocks != len(self.channels):
            raise ValueError(
                f"Expected depths and channels to have the same length, "
                f"but got {self.num_blocks} and {len(self.channels)}"
            )

        self.nets = nn.ModuleList(
            [self._build_path() for _ in range(self.num_paths)]
        )

    def _build_path(self):
        path = []
        c_in = self.in_channels
        for b, (d, c) in enumerate(zip(self.depths, self.channels)):
            blocks = []
            for i in range(d):
                stride = 2 if (i == 0 and b != self.num_blocks - 1) else 1
                blocks.append(
                    ConvModule(
                        in_channels=c_in if i == 0 else c,
                        out_channels=c,
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                        bias=False,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='ReLU'),
                    )
                )
                c_in = c
            path.append(nn.Sequential(*blocks))
        return nn.ModuleList(path)

    def _preprocess_input(self, x):
        h, w = x.shape[-2:]
        return [
            _interpolate(x, size=(int(r * h), int(r * w)))
            for r in self.scale_ratios
        ]

    def forward(self, x, interact_begin_idx=2):
        inputs = self._preprocess_input(x)
        feats = []
        # Initial processing up to interaction point
        for path, inp in zip(self.nets, inputs):
            x_path = inp
            for idx in range(interact_begin_idx + 1):
                x_path = path[idx](x_path)
            feats.append(x_path)

        # Multipath interactions and further processing
        for idx in range(interact_begin_idx + 1, self.num_blocks):
            feats = _multipath_interaction(feats)
            feats = [
                path[idx](feat) for path, feat in zip(self.nets, feats)
            ]

        # Upsample features to the same size
        size = feats[0].shape[-2:]
        feats = [_interpolate(feat, size=size) for feat in feats]

        # Concatenate features along the channel dimension
        out = torch.cat(feats, dim=1)

        # Return the concatenated feature map
        return [out]

    def init_weights(self):
        """Initialize the weights of the model."""
        super().init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)