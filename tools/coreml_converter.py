import torch
import torch.onnx
import warnings
import coremltools as ct

from mmseg.apis import init_model
from mmengine import Config

config_loc = 'configs/lpsnet/lpsnet_l_90k_4x4b_cityscapes-1536x768.py'
model_loc = 'work_dirs/lpsnet_l/iter_90000.pth'

def onnx_format_converter(config_file, model_file):
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, model_file, device='cuda:0')
    model.eval()
    
    dummy_input = torch.randn(1, 3, 1536, 512).cuda()
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model,
            dummy_input,
            "LPSNet_l.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )

def ml_convert():
    onnx_model_path = 'LPSNet_l.onnx'
    mlmodel = ct.converters.onnx.convert(model=onnx_model_path)
    mlmodel_fp16 = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=16)
    mlmodel_fp16.save("segmentation_model_fp16.mlmodel")

def main():
    ml_convert()

if __name__ == '__main__':
    main()
