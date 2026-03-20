"""UNet Decode Head with proper multi-scale skip connections.

This implements a symmetric decoder that mirrors the encoder's feature
hierarchy. At each resolution level, the decoder upsamples its output,
concatenates it with the corresponding encoder feature map (skip
connection), and refines through convolutional blocks.

Architecture:
    Encoder features:   F1(1/4)   F2(1/8)   F3(1/16)   F4(1/32)
                         │          │          │           │
    Decoder:             │          │          │        ┌──┘
                         │          │        ┌─┴──Up───Cat──Conv──┐
                         │        ┌─┴──Up───Cat──Conv──┘          │
                       ┌─┴──Up───Cat──Conv──┘                     │
                    Output (1/4)                                   │
                                                            Bottleneck
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


class ConvBnReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + skip connection (concatenation) + double convolution.

    This is the fundamental building block of the UNet decoder. It:
    1. Upsamples the deeper feature map by 2x (bilinear interpolation)
    2. Concatenates with the skip connection from the encoder
    3. Applies two 3x3 convolutions to refine the fused features

    Args:
        in_channels (int): Channels from the deeper level (to be upsampled).
        skip_channels (int): Channels from the encoder skip connection.
        out_channels (int): Output channels after fusion and convolution.
        dropout_ratio (float): Dropout ratio. Default: 0.0.
    """

    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, dropout_ratio: float = 0.0):
        super().__init__()
        # After concatenation: in_channels + skip_channels
        fused_channels = in_channels + skip_channels

        self.conv1 = ConvBnReLU(fused_channels, out_channels)
        self.conv2 = ConvBnReLU(out_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Feature from deeper level (B, C_deep, H, W).
            skip (Tensor): Encoder skip feature (B, C_skip, 2H, 2W).

        Returns:
            Tensor: Refined feature (B, C_out, 2H, 2W).
        """
        # Upsample to match skip connection spatial size
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=False)
        # Concatenate along channel dimension (skip connection)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


@HEADS.register_module()
class UNetDecodeHead(BaseDecodeHead):
    """UNet decoder head with multi-scale skip connections.

    Expects multi-scale encoder features at 4 resolution levels
    (e.g., 1/4, 1/8, 1/16, 1/32 for a ResNet backbone) and
    progressively upsamples from the deepest level, fusing with
    encoder features via skip connections at each level.

    The output resolution matches the highest encoder feature map
    (typically 1/4 of input). The segmentor's resize operation
    handles the final upsampling to full resolution.

    Args:
        in_channels (list[int]): Channel dims at each encoder scale.
            Must have exactly 4 elements, from shallowest to deepest.
            E.g., [256, 512, 1024, 2048] for ResNet.
        channels (int): Base channel count for decoder blocks.
            Decoder channels at each level: [channels, channels*2,
            channels*4], from shallowest to deepest.
        bottleneck_channels (int, optional): Channels for the bottleneck
            conv applied to the deepest encoder feature before decoding.
            If None, defaults to in_channels[-1].
        dropout_ratio (float): Dropout ratio in decoder blocks.
            Default: 0.1.
        num_classes (int): Number of output classes.
        norm_cfg (dict): Normalization config.
        align_corners (bool): Align corners for interpolation.
        loss_decode (dict): Loss config.
    """

    def __init__(self,
                 in_channels: List[int],
                 channels: int = 256,
                 bottleneck_channels: Optional[int] = None,
                 dropout_ratio: float = 0.1,
                 **kwargs):
        # BaseDecodeHead expects scalar in_channels; we pass the shallowest
        # level since the final output comes from that level.
        assert len(in_channels) == 4, \
            f'UNetDecodeHead requires exactly 4 encoder scales, got {len(in_channels)}'

        super().__init__(
            in_channels=in_channels[0],
            channels=channels,
            dropout_ratio=dropout_ratio,
            input_transform=None,
            in_index=[0, 1, 2, 3],
            **kwargs)

        # Store all encoder channel dims
        self.enc_channels = in_channels  # [C1, C2, C3, C4] shallow→deep
        c1, c2, c3, c4 = in_channels

        # Bottleneck: refine the deepest encoder features
        bneck_ch = bottleneck_channels or c4
        self.bottleneck = nn.Sequential(
            ConvBnReLU(c4, bneck_ch),
            ConvBnReLU(bneck_ch, bneck_ch),
        )

        # Decoder blocks (deep → shallow)
        # Level 3: upsample bottleneck, fuse with F3
        self.up3 = UpBlock(bneck_ch, c3, channels * 4, dropout_ratio)
        # Level 2: upsample level3 output, fuse with F2
        self.up2 = UpBlock(channels * 4, c2, channels * 2, dropout_ratio)
        # Level 1: upsample level2 output, fuse with F1
        self.up1 = UpBlock(channels * 2, c1, channels, dropout_ratio)

        # Override the cls_seg from BaseDecodeHead to match final channels
        self.conv_seg = nn.Conv2d(channels, self.num_classes, kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Decode with skip connections.

        Args:
            inputs (list[Tensor]): Encoder features [F1, F2, F3, F4]
                from shallowest (1/4) to deepest (1/32).

        Returns:
            Tensor: Segmentation logits (B, num_classes, H/4, W/4).
        """
        f1, f2, f3, f4 = inputs[0], inputs[1], inputs[2], inputs[3]

        # Bottleneck on deepest features
        x = self.bottleneck(f4)

        # Progressive upsampling with skip connections
        x = self.up3(x, f3)   # 1/32 → 1/16, fuse with F3
        x = self.up2(x, f2)   # 1/16 → 1/8,  fuse with F2
        x = self.up1(x, f1)   # 1/8  → 1/4,  fuse with F1

        # Classification
        out = self.conv_seg(x)
        return out
