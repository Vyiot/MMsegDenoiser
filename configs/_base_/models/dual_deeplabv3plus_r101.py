# Dual-Encoder DeepLabV3+ ResNet-101 (late fusion) for pseudo-label denoising.

num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='DualEncoderSegmentor',
    num_classes=num_classes,
    pretrained_rgb='open-mmlab://resnet101_v1c',
    pretrained_pl='open-mmlab://resnet101_v1c',
    backbone_rgb=dict(
        type='ResNetV1c',
        depth=101,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    backbone_pl=dict(
        type='ResNetV1c',
        depth=101,
        in_channels=3,  # Adapted to num_classes
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    fusion_neck=dict(
        type='FusionNeck',
        in_channels_rgb=[256, 512, 1024, 2048],
        in_channels_pl=[256, 512, 1024, 2048],
        out_channels=[256, 512, 1024, 2048],
        fusion_type='concat'),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
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
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
