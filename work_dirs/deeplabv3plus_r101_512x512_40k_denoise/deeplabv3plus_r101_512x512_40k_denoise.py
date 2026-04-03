num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DenoiserSegmentor',
    num_classes=2,
    adapt_input_conv=True,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'PseudoLabelDenoiseDataset'
data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'
img_suffix = ''
seg_map_suffix = ''
pseudo_label_suffix = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'img_info', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'flip_direction', 'img_norm_cfg', 'num_classes'
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'img_info', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'img_norm_cfg', 'num_classes'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'img_info', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
            'img_norm_cfg', 'num_classes'
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PseudoLabelDenoiseDataset',
        data_root='/home/ubuntu/vy/Denoiser/OEM_v2_Building',
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='train.txt',
        num_classes=2,
        img_suffix='',
        seg_map_suffix='',
        pseudo_label_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='PseudoLabelToOneHot', ignore_index=255),
            dict(type='FormatDenoiseBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=[
                    'img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                    'num_classes'
                ])
        ]),
    val=dict(
        type='PseudoLabelDenoiseDataset',
        data_root='/home/ubuntu/vy/Denoiser/OEM_v2_Building',
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='val.txt',
        num_classes=2,
        img_suffix='',
        seg_map_suffix='',
        pseudo_label_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='PseudoLabelToOneHot', ignore_index=255),
            dict(type='FormatDenoiseBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_semantic_seg'],
                meta_keys=[
                    'img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'img_norm_cfg', 'num_classes'
                ])
        ]),
    test=dict(
        type='PseudoLabelDenoiseDataset',
        data_root='/home/ubuntu/vy/Denoiser/OEM_v2_Building',
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='test.txt',
        num_classes=2,
        img_suffix='',
        seg_map_suffix='',
        pseudo_label_suffix='',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='PseudoLabelToOneHot', ignore_index=255),
            dict(type='FormatDenoiseBundle'),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'img_norm_cfg', 'num_classes'
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
device = 'cuda'
work_dir = './work_dirs/deeplabv3plus_r101_512x512_40k_denoise'
gpu_ids = range(0, 1)
