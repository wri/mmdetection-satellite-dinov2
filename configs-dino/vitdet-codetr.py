_base_ = [
   # '../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
    './tree-counting-ssj-scp-albu.py' # -ssj-scp
]

custom_imports = dict(imports=['projects.ViTDet.vitdet', 'projects.CO-DETR.codetr'])

backbone_norm_cfg = dict(type='LN', requires_grad=False)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/coco/'
base_lr = 3e-4
max_epochs = 100

# model settings
num_dec_layer = 4
loss_lambda = 1.0
num_classes = 1
rpn_weight = 0.9

image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
model = dict(
    type='CoDETR',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=False,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(type='DetDataPreprocessor', pad_size_divisor=32, batch_augments=batch_augments,
                          bgr_to_rgb = True, mean = [0.420 * 255, 0.411 * 255, 0.296 * 255],
                        std = [0.213 * 255, 0.156 * 255, 0.143 * 255]),
    backbone=dict(
        type='SSLVisionTransformer',
        img_size=image_size,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        #drop_path_rate=0.1,
        #window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        pretrained = None,
        out_indices=[
            9, 16, 22, 29,
        ],
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/ubuntu/mmdetection/models/SSLhuge_satellite.pth')),
    neck=dict(
        type='FPN_ViT',
        backbone_channel=1280,
        in_channels=[320, 640, 1280, 1280],
        out_channels=256,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    query_head=dict(
        type='CoDINOHead',
        num_query=2000,
        num_classes=num_classes,
        in_channels=1024,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=1,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=4,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=4,
                        dropout=0.0),
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=4,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4,
                            dropout=0.0),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
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
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
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
                    neg_iou_thr=0.5,
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
        rpn_proposal=dict(nms_pre = 12500, max_per_img=2500, nms=dict(iou_threshold=0.15)),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=768,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        #dict(
        #    assigner=dict(type='ATSSAssigner', topk=9),
        #    allowed_border=-1,
        #    pos_weight=-1,
        #    debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=2000,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='nms', iou_threshold=0.15)),
        dict(
            rpn=dict(
                nms_pre=10000,
                max_per_img=2500,
                nms=dict(type='nms', iou_threshold=0.15),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.15),
                max_per_img=2500)),
        #dict(
            # atss bbox head:
        #    nms_pre=10000,
        #    min_bbox_size=0,
        #    score_thr=0.0,
        #    nms=dict(type='nms', iou_threshold=0.1),
        #    max_per_img=2500),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=90,
        by_epoch=True,
        milestones=[40, 60, 75],
        gamma=0.5)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

train_cfg = dict(type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=4)

test_cfg = dict(type='TestLoop')

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [dict(type='Fp16CompresssionHook')]
auto_scale_lr = dict(enable=False, base_batch_size=16)