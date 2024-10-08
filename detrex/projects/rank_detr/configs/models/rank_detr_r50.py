import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.rank_detr.modeling import (
    RankDETR,
    RankDetrTransformerEncoder,
    RankDetrTransformerDecoder,
    RankDetrTransformer,
    RankDetrCriterion,
    HighOrderMatcher,
)

model = L(RankDETR)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(RankDetrTransformer)(
        encoder=L(RankDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=4,
            feedforward_dim=1024,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            use_checkpoint=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(RankDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=4,
            feedforward_dim=1024,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
            look_forward_twice=True,
            num_queries_one2one="${..num_queries_one2one}",
            num_queries_one2many="${..num_queries_one2many}",
            two_stage_num_proposals="${..two_stage_num_proposals}",
            rank_adaptive_classhead="${..rank_adaptive_classhead}",
            query_rank_layer=True,
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=4,
        num_queries_one2one="${..num_queries_one2one}",
        num_queries_one2many="${..num_queries_one2many}",
        two_stage_num_proposals=2250,
        mixed_selection=True,
        rank_adaptive_classhead="${..rank_adaptive_classhead}",
    ),
    embed_dim=256,
    num_classes=1,
    num_queries_one2one=750,
    num_queries_one2many=1500,
    aux_loss=True,
    with_box_refine=False,
    as_two_stage=False,
    criterion=L(RankDetrCriterion)(
        num_classes=1,
        matcher=L(HighOrderMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
            iou_order_alpha=4.0,
            matcher_change_iter=67500,
        ),
        weight_dict={
            "loss_class": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        GIoU_aware_class_loss=True,
    ),
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    select_box_nums_for_evaluation=750,
    device="cuda",
    mixed_selection=True,
    k_one2many=6,
    lambda_one2many=1.0,
    rank_adaptive_classhead=True,
)

# set aux loss weight dict
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict
