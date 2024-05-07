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

load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
   # dict(
    #    type='RandomResize',
    #    scale=image_size,
    #    ratio_range=(1, 1.25),
    #    keep_ratio=True),
    #dict(
    #    type='RandomCrop',
    #    crop_type='absolute_range',
    #    crop_size=image_size,
    ##    recompute_bbox=True,
    #    allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug', hue_delta=3, saturation_delta=15, value_delta=15),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='Pad', size=image_size),
]

train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=10, paste_by_box = True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5)),
    dict(type='PackDetInputs')
]


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='RandomCrop', crop_size=(384, 384), allow_negative_crop = True, recompute_bbox = True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    #dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(type='RepeatDataset',times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            pipeline=load_pipeline,
            filter_cfg=dict(filter_empty_gt=False, min_size=1),
            backend_args=None)
    ),
    pipeline = train_pipeline)

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=4,
    num_workers=0,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json',
        pipeline=test_pipeline,
        backend_args=None)
    )

val_dataloader = test_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train.json',
    metric=['bbox'],
    format_only=False)
test_cfg = dict(type='TestLoop')
val_cfg = dict(type='ValLoop')
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'), dict(type='DetVisualizationHook', draw = True, interval = 1, show = True)])
