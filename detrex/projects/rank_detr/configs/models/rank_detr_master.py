from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .rank_detr_r50 import model

from projects.stabledino.modeling import (
    SSLVisionTransformer
)

from detrex.modeling.neck import ChannelMapper


# ViT Base Hyper-params
embed_dim, depth, num_heads, dp = 1024, 24, 16, 0.2

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SSLVisionTransformer)(
        img_size=(512, 512),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.3,
        mlp_ratio=4,
        pretrained=None,
        frozen_stages = -1,
        qkv_bias=True,
        init_cfg=None,#dict(type='Pretrained', checkpoint='/home/ubuntu/mmdetection/models/SSLLarge.pth'),
        out_indices=[4, 11, 17, 23],
    )

model.pixel_mean = [107.2, 104.8, 75.38]
model.pixel_std = [54.21, 39.81, 36.52],
model.with_box_refine = True
model.as_two_stage = True
model.rank_adaptive_classhead = True
model.transformer.decoder.query_rank_layer = True
model.criterion.GIoU_aware_class_loss = True
model.criterion.matcher.iou_order_alpha = 4.0
model.criterion.matcher.matcher_change_iter = 40000

model.neck =L(ChannelMapper)(
        input_shapes={
            "p2": ShapeSpec(channels=256),
            "p3": ShapeSpec(channels=512),
            "p4": ShapeSpec(channels=1024),
            "p5": ShapeSpec(channels=1024),
        },
        in_features=["p2", "p3", "p4", "p5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        bias = False,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
)