# UNet with ResNet-50 encoder for pseudo-label denoising.
# Input channels = 3 (RGB) + num_classes (one-hot pseudo-label).
#
# Uses UNetDecodeHead with proper multi-scale skip connections:
#   F1(1/4) ──skip──┐
#   F2(1/8) ──skip──┤
#   F3(1/16)──skip──┤
#   F4(1/32)──bottleneck──Up+Cat+Conv──Up+Cat+Conv──Up+Cat+Conv──Output(1/4)

num_classes = 7

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DenoiserSegmentor',
    num_classes=num_classes,
    adapt_input_conv=True,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,  # Will be adapted to 3 + num_classes
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[256, 512, 1024, 2048],  # ResNet-50 feature dims
        channels=256,                          # Base decoder channel count
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
