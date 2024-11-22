import torch
import coremltools as ct
import torch.onnx
from mmseg.apis import init_model
from mmengine import Config

def convert_to_coreml(config_file, model_file):
    # Load the configuration
    cfg = Config.fromfile(config_file)
    
    # Initialize the model on CPU
    model = init_model(cfg, model_file, device='cpu')
    model.eval()
    
    # (Optional) If you have a state_dict to load
    # model.load_state_dict(torch.load('lpsnet_weights.pth', map_location='cpu'))

    # Trace the model with a sample input on CPU
    dummy_input = torch.rand(1, 3, 1536, 768)  # On CPU by default
    traced_model = torch.jit.trace(model, dummy_input)

    input_shape = (1, 3, 1536, 768)
    
    mlmodel = ct.convert(
        traced_model,  # or scripted_model
        inputs=[ct.ImageType(shape=input_shape, scale=1/255.0, bias=[-0.485, -0.456, -0.406], color_layout="RGB")],
    )

    mlmodel.save("LPSNet.mlmodel")

def main():
    config_loc = 'configs/lpsnet/lpsnet_l_90k_4x4b_cityscapes-1536x768.py'
    model_loc = 'work_dirs/lpsnet_l/iter_90000.pth'
    convert_to_coreml(config_loc, model_loc)

if __name__ == '__main__':
    main()
