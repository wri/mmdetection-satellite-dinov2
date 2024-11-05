# Inherit and overwrite part of the config based on this config
#_base_ = './rtmdet_tiny_8xb32-300e_coco.py'
_base_ = [
    #../../../default/_base_/schedules/schedule_20e.py',
    '../../../configs/_base_/default_runtime.py',
]

dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/coco/'
image_size = (512, 512)

#backend_args = None

metainfo = {
    'classes': ('tree'),
    'palette': [
        (220, 20, 60),
    ] 
}
# 40% defocus, 55% fog


albu_train_transforms = [
    #dict(
    #    type='ShiftScaleRotate',
    #    shift_limit=0.0,
    #    scale_limit=(1, 1.5),
    #    rotate_limit=0,
    #    interpolation=1,
    #    p=0.5),
    
    dict(
        type='OneOf', #60% clear image, 20% haze, 20% shadow
        transforms=[
            dict(type='Sequential',# Simulating haze - strong fog, strong defocus, strong B/C
                 transforms = [
                     dict(type='RandomFog',
                         p=0.55,
                         fog_coef_lower=0.15,
                         fog_coef_upper=0.8,
                         alpha_coef=0.1),
                     dict(
                        type='Defocus',
                        p=0.8,
                        radius=(2,5),
                        alias_blur=(0.1, .5)),
                     dict( 
                        type='RandomBrightnessContrast',
                        brightness_limit=[.05, .25],
                        contrast_limit=[-0.25, -0.05],
                        p=0.6
                     )], p = 0.05), # 0.2
            dict(type='Sequential', # Simulating mild B/C, mild Fog, mild defocus
                 transforms = [
                    dict(type='RandomFog',
                         p=0.4,
                         fog_coef_lower=0.1,
                         fog_coef_upper=0.7,
                         alpha_coef=0.1),
                    dict(
                        type='Defocus',
                        p=0.25,
                        radius=(2,4),
                        alias_blur=(0.1, .4)),
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[-.2, .2], #-.4, .4
                        contrast_limit=[-.2, .2], # -.4, .4
                        p=1.)
                 ], p = 0.25), # 0.7
            dict(type='Sequential', # Simulating cloud shadow
                 transforms = [
                    dict(
                        type='Defocus',
                        p=0.3,
                        radius=(2,3),
                        alias_blur=(0.1, .5)),
                    dict( # Simulating cloud shadow
                        type = 'RandomGamma',
                        gamma_limit=(150, 195),
                        p = 0.1) # .1
                 ], p = 0.01), # 0.1
            ], p = 0.25), # 0.7
    dict(
        type='HueSaturationValue',
        hue_shift_limit=8, #15
        sat_shift_limit=12, #25
        val_shift_limit=12, #25
        p=0.5), # 0.6
    dict(
        type='GaussNoise',
        always_apply=False,
        p=0.5,
        var_limit = (10, 200),
        per_channel = False,
        mean = 0)
]
load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-3, 1e-3)),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
       },
        skip_img_without_anno=False),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.75, 1.25),
        keep_ratio=True),
    dict(type='Pad', size=image_size),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    #dict(type='YOLOXHSVRandomAug', hue_delta=3, saturation_delta=10, value_delta=10),
    dict(type='RandomFlip', prob=0.5, direction = 'horizontal'),
    dict(type='RandomFlip', prob=0.5, direction = 'vertical'),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(5e-5, 5e-5)),
    #dict(type='LimitBBoxes', max_bboxes = 1500),
    dict(type='PackDetInputs')
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='RepeatDataset',times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            pipeline=load_pipeline,
            filter_cfg=dict(filter_empty_gt=False, min_size=0.25),
            backend_args=None)
    )

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Pad', size=image_size),
    #dict(type='RandomCrop', crop_size=(512, 512), allow_negative_crop = True, recompute_bbox = True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    #dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]



test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json',
        pipeline=test_pipeline,
        backend_args=None)
    )

val_dataloader = test_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox'],
    format_only=False)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')
test_evaluator = val_evaluator

default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=15),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)

log_level = 'DEBUG'
resume = False

#auto_scale_lr = dict(enable=False, base_batch_size=16)


visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'), dict(type='DetVisualizationHook', draw = True, interval = 1, show = True)])
