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


train_pipeline = [
    dict(type='Mosaic', img_scale=image_size, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-image_size[0] // 2, -image_size[1] // 2)),
   #dict(
   #     type='RandomAffine',
   #     scaling_ratio_range=(0.5, 1.5),
   #     # img_scale is (width, height)
   #     border=(-image_size[0] // 2, -image_size[1] // 2)),
    dict(
        type='MixUp',
        img_scale=image_size,
        ratio_range=(0.5, 1.5),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
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
    dataset=dict(type='RepeatDataset',times=4,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            data_prefix=dict(img='train/'),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_cfg=dict(filter_empty_gt=False, min_size=0),
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
