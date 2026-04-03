# SegFormer MiT-B2 with UNetDecodeHead (progressive skip connections).
#
# Replaces the flat MLP SegformerHead with UNetDecodeHead that performs
# progressive upsampling with skip connections at each encoder scale.
# This provides stronger spatial detail preservation for boundary
# correction in pseudo-label denoising.

num_classes = 2

model = dict(
    type='DenoiserSegmentor',
    num_classes=num_classes,
    adapt_input_conv=True,
    pretrained='pretrain/mit_b2.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,  # Adapted to 3 + num_classes
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
        type='UNetDecodeHead',
        in_channels=[64, 128, 320, 512],  # MiT-B2 feature dims
        channels=128,                      # Lighter decoder for SegFormer
        bottleneck_channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
