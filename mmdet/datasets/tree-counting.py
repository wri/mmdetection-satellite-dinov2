# Inherit and overwrite part of the config based on this config
#_base_ = './rtmdet_tiny_8xb32-300e_coco.py'
_base_ = [
    #'vitdet_testing.py'
    #../../../default/_base_/schedules/schedule_20e.py',
    #'../../../configs/_base_/default_runtime.py',
]
data_root = 'mmdetection/data/tree/' # dataset root
dataset_type = 'CocoDataset'

train_batch_size_per_gpu = 4
train_num_workers = 0

max_epochs = 20
stage2_num_epochs = 1
base_lr = 1e-5
image_size = (512, 512)


metainfo = {
    'classes': ('tree'),
    'palette': [
        (220, 20, 60),
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask = True),
    dict(
        type='Resize',
        scale = image_size,
        #scale=(640, 640),
        #ratio_range=(1., 1.),
        keep_ratio=True),
    #dict(type='RandomCrop', crop_size=(640, 640)),
    #dict(type='YOLOXHSVRandomAug'),
    #dict(type='RandomFlip', prob=0.5),
    #dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        #metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json',
        pipeline=train_pipeline)
    )


test_dataloader = train_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'train.json',
    metric=['bbox'],
    format_only=False)


test_evaluator = val_evaluator
#visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
#    dict(type='TensorboardVisBackend')])
