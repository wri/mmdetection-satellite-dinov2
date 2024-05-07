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
    dict(type='LoadImageFromFile', backend_args= None),
    #dict(
    #    type='Resize',
    #    scale = image_size,
        #scale=(640, 640),
        #ratio_range=(1., 1.),
    #    keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='RandomCrop', crop_size=(448, 448), allow_negative_crop = True, recompute_bbox = True),
    #dict(type='RandomCrop', crop_size=[(400, 400), (384, 384), (416, 416), (432, 432), (368, 368), (448, 448), (368, 368)],
    #     allow_negative_crop = True),
    #dict(type='Resize', scale = image_size, keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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



train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            data_root=data_root,
            type=dataset_type,
            metainfo=metainfo,
            data_prefix=dict(img='train/'),
            ann_file='train.json',
            pipeline=train_pipeline,
            backend_args=None)
    )
    )

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
#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
#    dict(type='TensorboardVisBackend')])
