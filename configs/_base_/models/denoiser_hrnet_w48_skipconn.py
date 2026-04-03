# HRNet-W48 with UNetDecodeHead (progressive skip connections).
#
# Replaces the flat resize_concat + 1x1 conv FCNHead with UNetDecodeHead
# that progressively upsamples from the deepest branch (384-ch, 1/32)
# and fuses with shallower branches via skip connections.
#
# HRNet already maintains multi-resolution features, but the original
# FCNHead simply resizes all 4 branches to the same resolution and
# concatenates them — losing the hierarchical structure. UNetDecodeHead
# respects the natural coarse-to-fine hierarchy.

num_classes = 2

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DenoiserSegmentor',
    num_classes=num_classes,
    adapt_input_conv=True,
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
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
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[48, 96, 192, 384],  # HRNet-W48 branch dims
        channels=96,                      # Lighter decoder for HRNet
        bottleneck_channels=384,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
