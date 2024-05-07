_base_ = [
    '../../../configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_coco.py',
    #'../../../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    #'./tree-verification.py',
    './tree-counting.py'
]


custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=False)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/tree/'
base_lr = 1e-4
max_epochs = 100

# model settings
model = dict(
    type = 'CenterNet',
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments,
                          bgr_to_rgb = True, mean = [0.420 * 255, 0.411 * 255, 0.296 * 255],
                        std = [0.213 * 255, 0.156 * 255, 0.143 * 255]),
    backbone=dict(
        type='SSLVisionTransformer',
        img_size=512,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        #drop_path_rate=0.1,
        #window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        pretrained = None,
        #pretrained = '../HighResCanopyHeight-main/saved_checkpoints/SSLhuge_satellite.pth',
        #norm_cfg=backbone_norm_cfg,
        out_indices=[
            9, 16, 22, 29,
        ],
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/ubuntu/mmdetection/models/SSLhuge_satellite.pth')),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=1280,
        in_channels=[320, 640, 1280, 1280],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='CenterNetUpdateHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='GaussianFocalLoss',
            pos_weight=0.25,
            neg_weight=0.75,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=10000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=3000)
)
   # rpn_head=dict(num_convs=2),
   # roi_head=dict(
   #     bbox_head=dict(
   #         type='Shared4Conv1FCBBoxHead',
   #         conv_out_channels=256,
   #         norm_cfg=norm_cfg),
   #     mask_head=dict(norm_cfg=norm_cfg)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.005),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

train_cfg = dict(type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=2)

test_cfg = dict(
        nms_pre=20000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1000)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
#custom_hooks = [dict(type='Fp16CompresssionHook')]
