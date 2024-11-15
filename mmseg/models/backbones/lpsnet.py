"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
import time

def upsample(x, size):
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    else:
        return x

def bi_interaction(x_h, x_l):
    sizeH = (int(x_h.shape[-2]), int(x_h.shape[-1]))
    sizeL = (int(x_l.shape[-2]), int(x_l.shape[-1]))
    o_h = x_h + upsample(x_l, sizeH)
    o_l = x_l + upsample(x_h, sizeL)
    return o_h, o_l

def tr_interaction(x1, x2, x3):
    s1 = (int(x1.shape[-2]), int(x1.shape[-1]))
    s2 = (int(x2.shape[-2]), int(x2.shape[-1]))
    s3 = (int(x3.shape[-2]), int(x3.shape[-1]))
    o1 = x1 + upsample(x2, s1) + upsample(x3, s1)
    o2 = x2 + upsample(x1, s2) + upsample(x3, s2)
    o3 = x3 + upsample(x2, s3) + upsample(x1, s3)
    return o1, o2, o3

class ConvBNReLU3x3(BaseModule):
    def __init__(self, c_in, c_out, stride, deploy=False, init_cfg=None):
        super(ConvBNReLU3x3, self).__init__(init_cfg)
        if deploy:
            self.conv = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=True)
            self.bn = None
        else:
            self.conv = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
            self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class BaseNet(BaseModule):
    def __init__(self, layers, channels, deploy=False, init_cfg=None):
        super(BaseNet, self).__init__(init_cfg)
        self.layers = layers
        assert len(self.layers) == 5
        self.channels = channels
        assert len(self.channels) == 5
        self.strides = (2, 2, 2, 2, 1)
        self.stages = nn.ModuleList()
        c_in = 3
        for l, c, s in zip(self.layers, self.channels, self.strides):
            self.stages.append(self._make_stage(c_in, c, l, s, deploy))
            c_in = c

    @staticmethod
    def _make_stage(c_in, c_out, numlayer, stride, deploy):
        layers = []
        for i in range(numlayer):
            layers.append(ConvBNReLU3x3(c_in if i == 0 else c_out, c_out, stride if i == 0 else 1, deploy))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        outputs = []
        for s in self.stages:
            out = s(out)
            outputs.append(out)
        return outputs

@MODELS.register_module()
class LPSNet(BaseModule):
    def __init__(self, depth, width, resolution, deploy=False, init_cfg=None):
        super(LPSNet, self).__init__(init_cfg)
        self.depth = depth
        assert len(self.depth) == 5
        self.width = width
        assert len(self.width) == 5
        self.resolution = resolution
        self.deploy = deploy

        self.resolution_filter = [r for r in resolution if r > 0]
        self.resolution_sorted = sorted(self.resolution_filter, reverse=True)
        self.num_paths = len(self.resolution_sorted)
        assert self.num_paths in [1, 2, 3], 'Only support 1, 2, or 3 paths'

        if self.num_paths == 1:
            self.net = BaseNet(self.depth, self.width, deploy)
        elif self.num_paths == 2:
            self.netH = BaseNet(self.depth, self.width, deploy)
            self.netL = BaseNet(self.depth, self.width, deploy)
        elif self.num_paths == 3:
            self.net1 = BaseNet(self.depth, self.width, deploy)
            self.net2 = BaseNet(self.depth, self.width, deploy)
            self.net3 = BaseNet(self.depth, self.width, deploy)
        else:
            raise NotImplementedError

    def _preprocess_input(self, x):
        r_list = self.resolution_sorted
        x_list = [upsample(x, (int(x.shape[-2] * r), int(x.shape[-1] * r))) for r in r_list]
        return x_list

    def forward(self, x):
        if self.num_paths == 1:
            x_processed = self._preprocess_input(x)[0]
            outs = self.net(x_processed)
            return [outs[-1]]  # Return only the last feature
        elif self.num_paths == 2:
            #print('Image Input')
            #print('Image size: ', x.size())
            #time.sleep(10)
            xh, xl = self._preprocess_input(x)
            #print('Input processor')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = self.netH.stages[0](xh), self.netL.stages[0](xl)
            #print('Stage 1')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = self.netH.stages[1](xh), self.netL.stages[1](xl)
            #print('Stage 2')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = self.netH.stages[2](xh), self.netL.stages[2](xl)
            #print('Stage 3')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = bi_interaction(xh, xl)
            #print('Interaction 1')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = self.netH.stages[3](xh), self.netL.stages[3](xl)
            #print('Stage 4')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = bi_interaction(xh, xl)
            #print('Interaction 2')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            xh, xl = self.netH.stages[4](xh), self.netL.stages[4](xl)
            #print('Stage 5')
            #print('High Resolution size: ', xh.size())
            #print('Low Resolution size: ', xl.size())
            #time.sleep(10)
            x_cat = torch.cat([xh, upsample(xl, (int(xh.shape[-2]), int(xh.shape[-1])))], dim=1)
            print('Image Output')
            print('Image size: ', x_cat.size())
            time.sleep(10)
            return [x_cat]  # Return the concatenated feature
        elif self.num_paths == 3:
            x1, x2, x3 = self._preprocess_input(x)
            x1, x2, x3 = self.net1.stages[0](x1), self.net2.stages[0](x2), self.net3.stages[0](x3)
            x1, x2, x3 = self.net1.stages[1](x1), self.net2.stages[1](x2), self.net3.stages[1](x3)
            x1, x2, x3 = self.net1.stages[2](x1), self.net2.stages[2](x2), self.net3.stages[2](x3)
            x1, x2, x3 = tr_interaction(x1, x2, x3)
            x1, x2, x3 = self.net1.stages[3](x1), self.net2.stages[3](x2), self.net3.stages[3](x3)
            x1, x2, x3 = tr_interaction(x1, x2, x3)
            x1, x2, x3 = self.net1.stages[4](x1), self.net2.stages[4](x2), self.net3.stages[4](x3)
            x_cat = [x1,
                     upsample(x2, (int(x1.shape[-2]), int(x1.shape[-1]))),
                     upsample(x3, (int(x1.shape[-2]), int(x1.shape[-1])))]
            x_cat = torch.cat(x_cat, dim=1)
            return [x_cat]  # Return the concatenated feature
        else:
            raise NotImplementedError
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

__all__ = [
    "LPSNet",
]

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