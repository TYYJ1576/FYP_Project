# mmseg/models/decode_heads/lpsnet_head.py

#import torch
#import torch.nn as nn
#from mmseg.registry import MODELS
#from mmseg.models.decode_heads.decode_head import BaseDecodeHead

#@MODELS.register_module()
#class LPSNetHead(BaseDecodeHead):
#    def __init__(self, in_channels, num_classes, init_cfg=None, **kwargs):
#        super(LPSNetHead, self).__init__(in_channels=in_channels,
#                                         channels=in_channels,
#                                         num_classes=num_classes,
#                                         input_transform=None,
#                                         init_cfg=init_cfg,
#                                         **kwargs)
#        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

#    def forward(self, inputs):
#        """Forward function."""
#        x = inputs[0]  # Get the feature from the backbone
#       print(x.size())
#        x = self.cls_seg(x)
#        print(x.size())
#        return x

# mmseg/models/decode_heads/lpsnet_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@MODELS.register_module()
class LPSNetHead(BaseDecodeHead):
    def __init__(self, in_channels, num_classes, init_cfg=None, **kwargs):
        super(LPSNetHead, self).__init__(in_channels=in_channels,
                                         channels=in_channels,
                                         num_classes=num_classes,
                                         input_transform=None,
                                         init_cfg=init_cfg,
                                         **kwargs)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)  # Transform the input as needed
        x = self.cls_seg(x)  # Apply the segmentation layer

        # Upsample the output to match the input image size
        output = F.interpolate(x, size=inputs[0].shape[-2:], mode='bilinear', align_corners=False)
        print(output.size())
        return output

