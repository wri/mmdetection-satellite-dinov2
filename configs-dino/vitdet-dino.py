_base_ = [
    '../../../configs/dino/dino-4scale_r50_8xb2-12e_coco.py',
    #'../../../configs/dino/dino-5scale.py',
    './tree-counting-ssj-scp-albu.py' # -ssj-scp
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
base_lr = 3e-4
max_epochs = 100

# model settings
model = dict(
    type='DINO',
    data_preprocessor=dict(type='DetDataPreprocessor', pad_size_divisor=32, batch_augments=batch_augments,
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
        type='FPN_ViT',
        backbone_channel=1280,
        in_channels=[320, 640, 1280, 1280],
        out_channels=256,
        norm_cfg=dict(type='GN', num_groups=32), # The DINO Authors use GN 32 on the input to the DINO encoder, so we do here too
        num_outs=4),
    )#,

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0002,   # 0.0002 for DeformDETR
        weight_decay=0.0001), # 0002
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=2,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))


train_cfg = dict(type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=40)

test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=5e-7,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[10, 30, 60, 75],
        gamma=0.5)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [dict(type='Fp16CompresssionHook')]
