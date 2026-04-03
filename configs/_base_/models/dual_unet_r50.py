# Dual-Encoder UNet ResNet-50 (late fusion) for pseudo-label denoising.
#
# Two ResNet-50 encoders + FusionNeck + UNetDecodeHead with skip connections.

num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='DualEncoderSegmentor',
    num_classes=num_classes,
    pretrained_rgb='open-mmlab://resnet50_v1c',
    pretrained_pl='open-mmlab://resnet50_v1c',
    backbone_rgb=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    backbone_pl=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,  # Adapted to num_classes
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    fusion_neck=dict(
        type='FusionNeck',
        in_channels_rgb=[256, 512, 1024, 2048],
        in_channels_pl=[256, 512, 1024, 2048],
        out_channels=[256, 512, 1024, 2048],
        fusion_type='concat'),
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[256, 512, 1024, 2048],  # Fused feature dims
        channels=256,
        bottleneck_channels=1024,
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
