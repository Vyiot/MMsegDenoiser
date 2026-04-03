# SegFormer MiT-B2 backbone for pseudo-label denoising.
# Input channels = 3 (RGB) + num_classes (one-hot pseudo-label).

num_classes = 2

model = dict(
    type='DenoiserSegmentor',
    num_classes=num_classes,
    adapt_input_conv=True,
    pretrained='pretrain/mit_b2.pth',  # Download from SegFormer repo
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,  # Will be adapted to 3 + num_classes
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
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
