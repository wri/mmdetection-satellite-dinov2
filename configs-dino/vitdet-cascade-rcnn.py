_base_ = [
    '../configs/_base_/models/cascade-rcnn_r50_fpn.py',
    './tree-counting-ssj-scp.py'
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])
backbone_norm_cfg = dict(type='LN', requires_grad=False)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/coco/'
base_lr = 2e-4
max_epochs = 100

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments,
                          bgr_to_rgb = True, mean = [0.420 * 255, 0.411 * 255, 0.296 * 255],
                        std = [0.213 * 255, 0.156 * 255, 0.143 * 255]),
    backbone=dict(
        _delete_=True,
        type='SSLVisionTransformer',
        img_size=image_size,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        mlp_ratio=4,
        qkv_bias=True,
        pretrained = None,
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
        norm_cfg=dict(type='GN',num_groups=32, requires_grad=True)
    ))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.01),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

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