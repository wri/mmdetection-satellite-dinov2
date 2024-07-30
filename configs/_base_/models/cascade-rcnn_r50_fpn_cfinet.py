# model settings
rpn_weight = 0.9
model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
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
                    loss_weight=0.33 * rpn_weight)),
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
                    loss_weight=2. * rpn_weight))]),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                #loss_bbox = dict(type='CIoULoss', loss_weight=1.0)),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                #loss_bbox = dict(type='CIoULoss', loss_weight=1.0)),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                #loss_bbox = dict(type='CIoULoss', loss_weight=1.0)),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='DynamicAssigner',
                    low_quality_iou_thr=0.2, #.2
                    base_pos_iou_thr=0.25, # .25
                    neg_iou_thr=0.15), # .15
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
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5, # 5
                    neg_iou_thr=0.5, # 4
                    min_pos_iou=0.5, # 4
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=768,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6, # 55
                    neg_iou_thr=0.6, # 45
                    min_pos_iou=0.6, # 45
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=768,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7, # 6
                    neg_iou_thr=0.7, # 5
                    min_pos_iou=0.7, # 5
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=768,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    #test_cfg=dict(
    #    rpn=dict(
    #        nms_pre=1000,
    #        max_per_img=1000,
    #        nms=dict(type='nms', iou_threshold=0.7),
    #        min_bbox_size=0),
    #    rcnn=dict(
    #        score_thr=0.05,
    #        nms=dict(type='nms', iou_threshold=0.5),
    #        max_per_img=100)))#,
    test_cfg=dict(
        rpn=dict(
            nms_pre=100000,
            max_per_img=25000,
            nms=dict(type='nms', iou_threshold=0.08),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.08),
            max_per_img=25000)))
