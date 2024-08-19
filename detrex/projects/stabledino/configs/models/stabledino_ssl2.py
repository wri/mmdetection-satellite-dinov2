import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L
from detectron2.modeling.mmdet_wrapper import MMDetBackbone

from detrex.modeling.neck import ChannelMapper
#from detrex.modeling.backbone import SSLVisionTransformer
from detrex.layers import PositionEmbeddingSine
#from detrex.modeling.backbone.ssl import SSLVisionTransformer

from projects.stabledino.modeling import (
    DINO,
    SSLVisionTransformer,
    StableDINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    StableDINOCriterion,
    StableDINOHungarianMatcher
)

# python tools/train_net.py --config-file projects/stabledino/configs/stabledino_ssl2.py     --num-gpus 1     dataloader.train.total_batch_size=2     train.output_dir="./output/stabledino_ssl2_detectron"     train.test_with_nms=0.2

model = L(DINO)(
    backbone=L(SSLVisionTransformer)(
        img_size=(512, 512),
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.3,
        mlp_ratio=4,
        pretrained=None,
        frozen_stages = 0,
        qkv_bias=True,
        init_cfg=None,#dict(type='Pretrained', checkpoint='/home/ubuntu/mmdetection/models/SSLLarge.pth'),
        out_indices=[4, 11, 17, 23],
    ),
    position_embedding=L(PositionEmbeddingSine)( # This is different
        num_pos_feats=128,
        temperature=20, #10000
        normalize=True,
        offset=0.0, # =0.5
    ),
    neck=L(ChannelMapper)(
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
    ),
    transformer=L(DINOTransformer)(
        encoder=L(StableDINOTransformerEncoder)(
            embed_dim=256,
            num_heads=4,
            feedforward_dim=1024,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,#"${..num_feature_levels}",
            multi_level_fusion="dense-fusion"
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=4,
            feedforward_dim=1024,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,#"${..num_feature_levels}",
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=1,
    num_queries=1700,
    aux_loss=True,
    criterion=L(StableDINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(StableDINOHungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
            cec_beta=0.5,
        ),
        weight_dict={
            "loss_class": 6.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
        use_ce_loss_type="stable-dino",
        ta_alpha=0.0,
        ta_beta=2.0,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean = [107.2, 104.8, 75.38], 
    pixel_std = [54.21, 39.81, 36.52],
    device="cuda",
    gdn_k=2,
    select_box_nums_for_evaluation=1700,
    neg_step_type='none',
    no_img_padding=False,
    dn_to_matching_block=False,
)
#pixel_mean=[0.420 * 255, 0.411 * 255, 0.296 * 255],
#pixel_std=[0.213 * 255, 0.156 * 255, 0.143 * 255],
# # set aux loss weight dict
# base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
# if model.aux_loss:
#     weight_dict = model.criterion.weight_dict
#     aux_weight_dict = {}
#     aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
#     for i in range(model.transformer.decoder.num_layers - 1):
#         aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
#     weight_dict.update(aux_weight_dict)
#     model.criterion.weight_dict = weight_dict
