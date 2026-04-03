# Dataset config for pseudo-label denoising.
#
# Expected directory structure:
#   data_root/
#   ├── images/
#   │   ├── train/    (satellite RGB images)
#   │   └── val/
#   ├── pseudo_labels/
#   │   ├── train/    (noisy pseudo-labels as class index PNGs)
#   │   └── val/
#   └── clean_labels/
#       ├── train/    (ground-truth labels as class index PNGs)
#       └── val/
#
# Adjust `data_root`, `num_classes`, `classes`, `palette`,
# and file suffixes to match your dataset.

dataset_type = 'PseudoLabelDenoiseDataset'
data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'
# num_classes = 2
img_suffix = ''
seg_map_suffix = ''
pseudo_label_suffix = ''

# Normalization for satellite RGB images
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys=['img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                    'num_classes']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys=['img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'img_norm_cfg', 'num_classes']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPseudoLabel', to_onehot=False, ignore_index=255),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PseudoLabelToOneHot', ignore_index=255),
    dict(type='FormatDenoiseBundle'),
    dict(type='Collect', keys=['img'],
         meta_keys=['img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'img_norm_cfg', 'num_classes']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='train.txt',
        num_classes=2,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pseudo_label_suffix=pseudo_label_suffix,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='val.txt',
        num_classes=2,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pseudo_label_suffix=pseudo_label_suffix,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pseudo_label_dir='pseudolabels',
        reduce_zero_label=False,
        split='test.txt',
        num_classes=2,
        img_suffix=img_suffix,
        seg_map_suffix=seg_map_suffix,
        pseudo_label_suffix=pseudo_label_suffix,
        pipeline=test_pipeline))
