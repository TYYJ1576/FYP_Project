## Installation

#### Part 1: Installation of miniconda3
```bash
# Create a new folder call miniconda3
mkdir -p ~/miniconda3
# Download the installation package online
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
# Install the package
~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# Remove the installation package
rm ~/miniconda3/miniconda.sh
# Initialization of miniconda3
source miniconda3/bin/activate
conda init --all
```

#### Part 2: Initialization of miniconda environment
```bash
# Create a new miniconda environment
# python 3.11.8 is suggested. Current Date = 8-11-2024
conda create -n "openmmlab" python=3.11.8
# Check cuda version
nvidia-smi
# Install cuda toolkit
# install the version according to the cuda version of your machine
conda install nvidia/label/cuda-11.7.0::cuda-toolkit
```

#### Part 3: Installation of PyTorch
```bash
# Install PyTorch
# Install the version according to the cuda version of your machine
# Find different version of PyTorch here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### Part 4: Installation of MMSegmentation and all related libraries
```bash
# Install libraries from MMOpenLab
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
# Clone mmsegmentation from github
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
# Install all libraries that required by mmsegmentation
pip install -v -e .
# Install some missing libraries
pip install ftfy
pip install regex
# Adjust the source code of mmsegmentation to adapt the new versions of mmcv
cd mmseg
vim __init__.py
i
# Change 
#   assert (mmcv_min_version <= mmcv_version < mmcv_max_version)
# To
#   assert (mmcv_min_version <= mmcv_version <= mmcv_max_version)
:wq
```

#### Part 5: Verification
```bash
# Download the network file and the pretrained model
mim download mmsegmentation --config pspnet_r50-d8_4xb2-40k_cityscapes-512x1024 --dest .
# Produce a test result image
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
```