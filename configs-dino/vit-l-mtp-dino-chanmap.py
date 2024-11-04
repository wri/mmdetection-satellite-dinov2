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
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=False)
]
dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/coco/'
base_lr = 3e-4
max_epochs = 100
# model settings
model = dict(
    type='DINO',
    num_queries=1600,
    data_preprocessor=dict(type='DetDataPreprocessor', pad_size_divisor=32, batch_augments=batch_augments,
                          bgr_to_rgb = True, mean = [107.2, 104.8, 75.38],
                        std = [54.21, 39.81, 36.52]),
    backbone=dict(
        _delete_=True,
        type='SSLVisionTransformer',
        img_size=image_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        drop_path_rate=0.3,
        #window_size=14,
        mlp_ratio=4,
        frozen_stages = -1,
        qkv_bias=True,
        pretrained = None,
        #pretrained = '../HighResCanopyHeight-main/saved_checkpoints/SSLhuge_satellite.pth',
        #norm_cfg=backbone_norm_cfg,
        out_indices=[
            4, 11, 17, 23
        ],
        init_cfg=dict(
            type='Pretrained', checkpoint='/home/ubuntu/mmdetection/models/SSLLarge.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    #neck=dict(
    #    type='FPN_ViT',
    #    backbone_channel=1024,
    #    in_channels=[256, 512, 1024, 1024],
    #    out_channels=256,
        #norm_cfg=dict(type='GN', num_groups=32), # The DINO Authors use GN 32 on the input to the DINO encoder, so we do here too
    #    num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 1024 for DeformDETR, MASTER MODEL
                ffn_drop=0.))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=4, ### THIS IS 4 in  the  MASTER MODEL
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 1024 for DeformDETR,  MASTER MODEL
                ffn_drop=0.)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=200)),  # TODO: half num_dn_queries
    test_cfg=dict(max_per_img=1600, # MASTER: 1850!!!!
                 nms=dict(type='nms', iou_threshold=0.1))
    )


optim_wrapper = dict(
    optimizer=dict(
    type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.0001),
    #constructor='LayerDecayOptimizerConstructor_ViT', 
    accumulative_counts=4,
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(weight_decay = 0.01),
                                    'backbone.norm': dict(lr_mult=0.1, weight_decay = 0.0), # following best practice here
                                    'backbone.blocks': dict(lr_mult=0.1, weight_decay = 0.01),
                                    'backbone.pos_embed': dict(lr_mult=0.1, wd_mult = 0, weight_decay = 0),
                                    'backbone.cls_token': dict(lr_mult=0., wd_mult = 0, weight_decay = 0),
                                    'backbone.dist_token': dict(lr_mult=0.1, wd_mult = 0, weight_decay = 0),
                                    'backbone.mask_token': dict(lr_mult=0.0),
                                    'backbone.patch_embed': dict(lr_mult=0.1, weight_decay = 0.01),
                                   }))
    #paramwise_cfg=dict(
    #    num_layers=24, 
    #    layer_decay_rate=0.85,
    #    )
    #    )


train_cfg = dict(type='EpochBasedTrainLoop', 
    max_epochs=max_epochs, 
    val_interval=1)

test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-7,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[32],
        gamma=0.1)
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_hooks = [dict(type='Fp16CompresssionHook')]
