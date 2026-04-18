# Diffusion-aware denoiser: same architecture as the skipconn baseline
# but with timestep embedding in the decoder and multi-timestep training data.
#
# Changes vs baseline:
#   - decode_head: use_time_embd=True, num_timesteps=6
#   - diffusion_cfg: betas schedule for q_sample re-noising
#   - train dataset: use_all_t=True (loads noise masks, 12 samples/image)
#   - train pipeline: adds 'timesteps' to Collect keys
#   - DiffusionEvalHook: dual validation (single + iterative)

_base_ = [
    '../_base_/models/denoiser_deeplabv3plus_r101_skipconn.py',
    '../_base_/datasets/pseudo_label_denoise.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

total_iters = 26000
runner = dict(type='IterBasedRunner', max_iters=total_iters)
checkpoint_config = dict(interval=-1)

data_root = '/home/ubuntu/vy/Denoiser/OEM_v2_Building'

# Enable time embedding in decoder + diffusion schedule
model = dict(
    decode_head=dict(
        use_time_embd=True,
        num_timesteps=6),
    diffusion_cfg=dict(
        betas=dict(
            type='linear',
            start=0.8,
            stop=0,
            num_timesteps=6)))

# Override optimizer
optimizer = dict(lr=1e-4, weight_decay=0.0005)

# Train pipeline: same as base but add 'timesteps' to Collect keys
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
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'timesteps'],
         meta_keys=['img_info', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
                    'num_classes']),
]

# Override data: enable use_all_t for train, use custom pipeline
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        use_all_t=True,
        pipeline=train_pipeline))

# Config for DiffusionEvalHook (read by train.py)
diffusion_eval = dict(
    interval=1300,
    label_dir=data_root + '/labels',
    pseudo_dir=data_root + '/pseudolabels')

device = 'cuda'
