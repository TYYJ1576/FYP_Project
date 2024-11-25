from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
import math

# Create a new function based on F.interpolate but with some parameters fixed
# F.interpolate: Resizes or upscales/downscales a tensor
# mode="bilinear": 
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
class LPSNet_Block(BaseModule):

    def __init__(
        self,
        depths,
        channels,
        scale_ratios,
        in_channels=3,
        conv_type='separable',  # Add option to select convolution type
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
        self.conv_type = conv_type

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
                if self.conv_type == 'separable':
                    blocks.append(
                        SeparableConvModule(
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
                elif self.conv_type == 'standard':
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
                elif self.conv_type == 'residual':
                    blocks.append(
                        ResidualBlock(
                            in_channels=c_in if i == 0 else c,
                            out_channels=c,
                            stride=stride
                        )
                    )
                elif self.conv_type == 'bottleneck':
                    blocks.append(
                        BottleneckBlock(
                            in_channels=c_in if i == 0 else c,
                            out_channels=c,
                            stride=stride
                        )
                    )
                elif self.conv_type == 'ghost':
                    blocks.append(
                        GhostModule(
                            in_channels=c_in if i == 0 else c,
                            out_channels=c,
                            stride=stride
                        )
                    )
                elif self.conv_type == 'inverted_residual':
                    blocks.append(
                        InvertedResidualBlock(
                            in_channels=c_in if i == 0 else c,
                            out_channels=c,
                            stride=stride
                        )
                    )
                elif self.conv_type == 'shufflenet':
                    blocks.append(
                        ShuffleNetUnit(
                            in_channels=c_in if i == 0 else c,
                            out_channels=c,
                            stride=stride
                        )
                    )
                else:
                    raise ValueError(f"Unsupported convolution type: {self.conv_type}")
                c_in = c
            path.append(nn.Sequential(*blocks))
        return nn.ModuleList(path)

    def _preprocess_input(self, x):
        # Get the height and width from the input tensor
        h, w = x.shape[-2:]

        # Convert height and width to tensors
        h_tensor = torch.tensor(h, dtype=torch.float32)
        w_tensor = torch.tensor(w, dtype=torch.float32)

        # Return a list where each element is the resized version of 'x' using the scale ratios
        return [
            _interpolate(
                x,
                size=(
                    torch.floor(r * h_tensor).to(dtype=torch.int64),
                    torch.floor(r * w_tensor).to(dtype=torch.int64)
                )
            )
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


class SeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
    ):
        super(SeparableConvModule, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Groups set to in_channels for depthwise convolution
            bias=bias
        )

        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

        # Normalization layer
        if norm_cfg is not None:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        # Activation layer
        if act_cfg is not None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, ratio=2):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidualBlock, self).__init__()
        mid_channels = in_channels * expand_ratio

        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(mid_channels)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(mid_channels)
        self.project = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.expand(x)
        out = self.expand_bn(out)
        out = self.relu(out)

        out = self.depthwise(out)
        out = self.depthwise_bn(out)
        out = self.relu(out)

        out = self.project(out)
        out = self.project_bn(out)

        if self.use_res_connect:
            return x + out
        else:
            return out


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShuffleNetUnit, self).__init__()
        mid_channels = out_channels // 4

        self.stride = stride
        self.mid_channels = mid_channels

        if stride == 2:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels - in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels - in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch_main = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([x, self.branch_main(x)], dim=1)
        else:
            out = self.branch_main(x)
        return self.relu(out)
