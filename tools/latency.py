import torch
import random
import subprocess
import onnx
import os
from onnxsim import simplify
import warnings
from mmseg.models import build_segmentor
from mmengine.config import Config
from mmseg.apis import init_model
import time

warnings.filterwarnings('ignore')

@torch.no_grad()
def measure_latency_ms(model, data_in, device=0):
    def parse_latency(lines, pattern='[I] mean:'):
        lines = lines[::-1]
        for l in lines:
            if pattern in l:
                l_num = l[l.find(pattern) + len(pattern):l.find('ms')]
                return float(l_num.strip())
        return -1
    try:
        _ = subprocess.run('trtexec', stdout=subprocess.PIPE, check=True)
    except Exception:
        print("TensorRT's trtexec not found, falling back to PyTorch latency measurement.")
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):  # Run multiple iterations to get an average latency
            model(data_in)
        torch.cuda.synchronize()
        end_time = time.time()
        latency = (end_time - start_time) / 100 * 1000  # Convert to milliseconds
        return latency
    else:
        onnx_file_name = ''.join(random.sample('zyxwvutsrqponmlkjihgfedcba', 10)) + '.onnx'
        torch.onnx.export(model, (data_in,), onnx_file_name, verbose=False, opset_version=11)
        torch.cuda.empty_cache()
        model_simp, check = simplify(onnx.load(onnx_file_name))
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(model_simp, onnx_file_name)
        trtcmd = ['trtexec', '--workspace=2048', '--duration=20', '--warmUp=1000',
                  '--onnx=' + onnx_file_name, '--device={}'.format(device)]
        ret = subprocess.run(trtcmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        latency = parse_latency(ret.stdout.decode('utf-8').split('\n'))
        os.remove(onnx_file_name)
        return latency


def main():
    # Load configuration from MMSegmentation
    config_file = 'configs/lpsnet/lpsnet_s_90k_cityscapes-768x768.py'  # Adjust path accordingly
    checkpoint_file = 'work_dirs/test/iter_100.pth'  # Adjust path accordingly
    cfg = Config.fromfile(config_file)
    # Initialize model from config and checkpoint
    model = init_model(cfg, checkpoint_file, device='cuda:0').eval()
    # Evaluate inference latency with TensorRT or PyTorch fallback
    latency = measure_latency_ms(model, torch.rand((1, 3, 1024, 2048)).to('cuda'))
    print('Inference latency is {}ms/image on current device.'.format(latency))


if __name__ == '__main__':
    main()
