# the new config inherits the base configs to highlight the necessary modification
_base_ = 'cascade-mask-rcnn_r101_fpn_1x_coco.py'

# 1. dataset settings
# Inherit and overwrite part of the config based on this config
#_base_ = './rtmdet_tiny_8xb32-300e_coco.py'
_base_ = [
    #'vitdet_testing.py'
    #../../../default/_base_/schedules/schedule_20e.py',
    #'../../../configs/_base_/default_runtime.py',
]

dataset_type = 'CocoDataset'
data_root = 'mmdetection/data/tree/' # dataset root
image_size = (512, 512)

backend_args = None

metainfo = {
    'classes': ('tree'),
    'palette': [
        (220, 20, 60),
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args= backend_args),
    #dict(
    #    type='Resize',
    #    scale = image_size,
        #scale=(640, 640),
        #ratio_range=(1., 1.),
    #    keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='RandomCrop', crop_size=(640, 640)),
    #dict(type='YOLOXHSVRandomAug'),
    #dict(type='RandomFlip', prob=0.5),
    #dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    #dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json',
        pipeline=train_pipeline,
        backend_args=None)
    )

test_dataloader = dict(
    batch_size=1,
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


test_evaluator = val_evaluator
#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
#    dict(type='TensorboardVisBackend')])

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1)],
    # explicitly over-write all the `num_classes` field from default 80 to 5.
    mask_head=dict(num_classes=1)))