import torch
from torch.profiler import profile, record_function, ProfilerActivity
from mmseg.apis import init_model
from mmengine import Config

def main():
    # Initialize the MMSegmentation model
    config_file = 'configs/lpsnet/lpsnet_m_90k_cityscapes-768x768.py'
    checkpoint_file = 'iter_63000.pth'
    device = 'cuda:0'

    model = init_model(config_file, checkpoint_file, device=device)
    model.eval()

    # Load the model's config to get input dimensions
    cfg = Config.fromfile(config_file)
    img_scale = (768, 768)
    for pipeline_step in cfg.test_pipeline:
        if 'img_scale' in pipeline_step:
            img_scale = pipeline_step['img_scale']
            break
    if img_scale is None:
        img_scale = (1024, 512)
        print(f'img_scale not found in config. Using default img_scale: {img_scale}')
    if isinstance(img_scale, list):
        img_scale = img_scale[0]
    input_width, input_height = img_scale

    # Prepare input tensor
    input_tensor = torch.randn(1, 3, input_height, input_width).to(device)

    # Warm up the model
    with torch.no_grad():
        for _ in range(10):
            _ = model(return_loss=False, img=[input_tensor])

    # Measure latency using PyTorch profiler
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with torch.no_grad():
            _ = model(return_loss=False, img=[input_tensor])

    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

if __name__ == '__main__':
    main()