# Dual-Encoder SegFormer MiT-B2 (late fusion) + UNetDecodeHead skip connections.

num_classes = 7

model = dict(
    type='DualEncoderSegmentor',
    num_classes=num_classes,
    pretrained_rgb='pretrain/mit_b2.pth',
    pretrained_pl='pretrain/mit_b2.pth',
    backbone_rgb=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    backbone_pl=dict(
        type='MixVisionTransformer',
        in_channels=3,  # Adapted to num_classes
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    fusion_neck=dict(
        type='FusionNeck',
        in_channels_rgb=[64, 128, 320, 512],
        in_channels_pl=[64, 128, 320, 512],
        out_channels=[64, 128, 320, 512],
        fusion_type='concat'),
    decode_head=dict(
        type='UNetDecodeHead',
        in_channels=[64, 128, 320, 512],
        channels=128,
        bottleneck_channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
