_base_ = [
    'faster-rcnn_r50_fpn.py',
]

find_unused_parameters=True
rpn_weight = 0.9
model = dict(
    type='FasterRCNN',
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    rpn_head=dict(
        _delete_=True,
        type='CRPNHead',
        num_stages=2,
        num_classes=1,
        stages=[
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[2],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                refine_reg_factor=200.0,
                refine_cfg=dict(type='dilation', dilation=3),
                refined_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=1.0 * rpn_weight)),
            dict(
                type='StageRefineRPNHead',
                in_channels=256,
                feat_channels=256,
                refine_cfg=dict(type='offset'),
                refined_feature=True,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=4.0 * rpn_weight))]),
# model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='DynamicAssigner',
                    low_quality_iou_thr=0.3, #.2
                    base_pos_iou_thr=0.35, # .25
                    neg_iou_thr=0.25), # .15
                allowed_border=-1, 
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(nms_pre = 25000, max_per_img=10000, nms=dict(iou_threshold=0.1)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.50, neg_iou_thr=0.50, min_pos_iou=0.50),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5))),
    test_cfg=dict(
        rpn=dict(nms_pre = 25000, max_per_img=10000, nms=dict(iou_threshold=0.1)),
        rcnn=dict(score_thr=0.05, max_per_img=10000, nms=dict(type='nms', iou_threshold=0.1)))
)