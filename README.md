# mmdetection-satellite-dinov2
mmdetection-satellite-dinov2

[[`Paper`](https://doi.org/10.1016/j.rse.2023.113888)][[`ArxiV [same content]`](https://arxiv.org/abs/2304.07213)] [[`Blog`](https://research.facebook.com/blog/2023/4/every-tree-counts-large-scale-mapping-of-canopy-height-at-the-resolution-of-individual-trees/)] [[`BibTeX`](#citing-HighResCanopyHeight)]

# Overview

![](https://raw.githubusercontent.com/wri/mmdetection-satellite-dinov2/main/resources/qualitative-results.jpg)

This repository contains the (**WORK IN PROGRESS**) integration of the DiNOV2 SSL layer from [Meta and WRI](https://github.com/facebookresearch/HighResCanopyHeight) with MMDetection. MMdet 3.X has been updated to include the SSL layer as a backbone. Example configuration for training a ViTDet with a Cascade RCNN head on a COCO-style dataset can be found in the `configs-dino` folder.

# Getting started

```
conda create --name mmdet python=3.10 
conda activate mmdet
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install future tensorboard

pip install -U openmim
mim install mmcv-full

import torch
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())
```

# Training

```
# import torch
from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config, DictAction

config='path/to/config.py'
checkpoint = 'path/to/SSLhuge_satellite.pth'
cfg = Config.fromfile(config)
cfg.work_dir = osp.join(path/to/work/dir/')
runner = Runner.from_cfg(cfg)
runner.train()
```
