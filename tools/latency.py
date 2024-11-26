import torch
import time

from mmseg.apis import init_model
from mmengine import Config
from mmseg.models.backbones.lpsnet import LPSNet

config_file = 'work_dirs/conv_blocks/lpsnet_m_90k_4x4b_cityscapes-1536x768_Ghost/lpsnet_m_90k_4x4b_cityscapes-1536x768_Ghost.py'
checkpoint_file = 'work_dirs/conv_blocks/lpsnet_m_90k_4x4b_cityscapes-1536x768_Ghost/iter_90000.pth'
depth = [1, 3, 3, 10, 10]
width = [8, 24, 48, 96, 96]
resolution = [1, 1/4, 0]

def measure_latency_ms_1(config_file, depths, channels, scale_ratios):
    cfg = Config.fromfile(config_file)

    cfg.model.backbone.depths = depths
    cfg.model.backbone.channels = channels
    cfg.model.backbone.scale_ratios = scale_ratios

    path_count = len([i for i in scale_ratios if i > 0])
    cfg.model.decode_head.in_channels = channels[-1] * path_count
    cfg.model.decode_head.channels = channels[-1] * path_count

    model = init_model(cfg, checkpoint=None, device='cuda:0')
    model.eval()

    input_tensor = torch.randn(1, 3, 768, 1536).cuda()

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

    latency_acc = 0

    for _ in range(1000):
        start_time = time.time()

        with torch.no_grad():
            _ = model(input_tensor)

        end_time = time.time()
        latency_acc += (end_time - start_time)

    latency = (latency_acc / 1000) * 1000
    return latency

def measure_latency_ms_2(config_file, checkpoint_file):
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, checkpoint_file, device='cuda:0')
    model.eval()
    # batch size = 1, channels = 3, resolution = 768x1536
    input_tensor = torch.randn(1, 3, 768, 1536).cuda()


    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    latency_acc = 0

    with torch.no_grad():
        for _ in range(1000):
            start_time = time.time()

            _ = model(input_tensor)

            end_time = time.time()
            latency_acc += (end_time - start_time)
    
    latency = (latency_acc / 1000) * 1000
    print(f"Inference Latency: {latency:.3f} ms")

def main():
   # measure_latency_ms_1(config_file=config_file, depths=depth, channels=width, scale_ratios=resolution)
    measure_latency_ms_2(config_file, checkpoint_file)

if __name__ == '__main__':
    main()
