import torch
import time

from mmseg.apis import init_model
from mmengine import Config

config_file = 'configs/lpsnet/lpsnet_m_90k_4x1b_cityscapes-1536x768_PolyLR.py'
cfg = Config.fromfile(config_file)

checkpoint_file = 'work_dirs/LR/PolyLR/iter_90000.pth'

model = init_model(cfg, checkpoint_file, device='cuda:0')

model.eval()

input_tensor = torch.randn(1, 3, 768, 1536).cuda()

with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)

latency_acc = 0

for i in range(1000):
    start_time = time.time()

    with torch.no_grad():
        _ = model(input_tensor)

    end_time = time.time()
    latency_acc += (end_time - start_time)

latency = (latency_acc / 1000) * 1000
print(f"Inference Latency: {latency:.3f} ms")