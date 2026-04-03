# Dual-Encoder HRNet-W48 (late fusion) + UNetDecodeHead skip connections.

num_classes = 2
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='DualEncoderSegmentor',
    num_classes=num_classes,
    pretrained_rgb='open-mmlab://msra/hrnetv2_w48',
    pretrained_pl='open-mmlab://msra/hrnetv2_w48',
    backbone_rgb=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1, num_branches=1, block='BOTTLENECK',
                num_blocks=(4,), num_channels=(64,)),
            stage2=dict(
                num_modules=1, num_branches=2, block='BASIC',
                num_blocks=(4, 4), num_channels=(48, 96)),
            stage3=dict(
                num_modules=4, num_branches=3, block='BASIC',
                num_blocks=(4, 4, 4), num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3, num_branches=4, block='BASIC',
                num_blocks=(4, 4, 4, 4), num_channels=(48, 96, 192, 384)))),
    backbone_pl=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1, num_branches=1, block='BOTTLENECK',
                num_blocks=(4,), num_channels=(64,)),
            stage2=dict(
                num_modules=1, num_branches=2, block='BASIC',
                num_blocks=(4, 4), num_channels=(48, 96)),
            stage3=dict(
                num_modules=4, num_branches=3, block='BASIC',
                num_blocks=(4, 4, 4), num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3, num_branches=4, block='BASIC',
                num_blocks=(4, 4, 4, 4), num_channels=(48, 96, 192, 384)))),
    fusion_neck=dict(
        type='FusionNeck',
        in_channels_rgb=[48, 96, 192, 384],
        in_channels_pl=[48, 96, 192, 384],
        out_channels=[48, 96, 192, 384],
        fusion_type='concat'),
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[48, 96, 192, 384],
        channels=96,
        bottleneck_channels=384,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
