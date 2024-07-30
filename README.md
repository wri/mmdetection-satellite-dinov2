# mmdetection-satellite-dinov2
mmdetection-satellite-dinov2

[[`Paper`](https://doi.org/10.1016/j.rse.2023.113888)][[`ArxiV [same content]`](https://arxiv.org/abs/2304.07213)] [[`Blog`](https://research.facebook.com/blog/2023/4/every-tree-counts-large-scale-mapping-of-canopy-height-at-the-resolution-of-individual-trees/)] [[`BibTeX`](#citing-HighResCanopyHeight)]

# Overview

![](https://raw.githubusercontent.com/wri/mmdetection-satellite-dinov2/main/resources/qualitative-results.jpg)

This repository contains the (**WORK IN PROGRESS**) integration of the DiNOV2 SSL layer from [Meta and WRI](https://github.com/facebookresearch/HighResCanopyHeight) with MMDetection. MMdet 3.X has been updated to include the SSL layer as a backbone. Example configuration for training a ViTDet with a Cascade RCNN head on a COCO-style dataset can be found in the `configs-dino` folder.

# Getting started
An example notebook can be found [here](https://github.com/wri/mmdetection-satellite-dinov2/blob/main/train-model.ipynb)
