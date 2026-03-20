# DeepLabV3+ ResNet-101 with UNetDecodeHead (progressive skip connections).
#
# Replaces DepthwiseSeparableASPPHead (which only uses F1 + F4) with
# UNetDecodeHead that uses all 4 encoder scales with progressive
# upsampling and skip connections.
#
# Note: the dilated ResNet uses strides (1,2,1,1) with dilations (1,1,2,4),
# so the effective output strides are 1/4, 1/8, 1/8, 1/8 (F3 and F4 share
# the same spatial resolution). UNetDecodeHead handles this gracefully via
# bilinear interpolation matching in UpBlock.

num_classes = 7

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DenoiserSegmentor',
    num_classes=num_classes,
    adapt_input_conv=True,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        in_channels=3,  # Adapted to 3 + num_classes
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[256, 512, 1024, 2048],  # ResNet-101 feature dims
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
