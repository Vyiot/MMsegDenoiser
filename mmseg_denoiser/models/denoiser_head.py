"""DenoiserHead: a lightweight decode head with optional residual connection.

For pseudo-label denoising, a residual connection from the input pseudo-label
to the output can stabilize training. The network learns a correction
Δ = f(satellite, pseudo_label) and the final prediction is pseudo_label + Δ.
This is optional and controlled by `use_residual`.
"""

import torch
import torch.nn as nn
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class DenoiserHead(BaseDecodeHead):
    """Decode head with optional residual refinement for label denoising.

    When `use_residual=True`, the head predicts a correction logit that
    is added to the pseudo-label logit (converted from one-hot to logit
    space). This encourages the network to focus on correcting errors
    rather than predicting from scratch.

    Args:
        use_residual (bool): If True, add residual from pseudo-label.
            Default: False.
        residual_scale (float): Scale factor for the residual. Default: 1.0.
        **kwargs: Arguments for BaseDecodeHead.
    """

    def __init__(self, use_residual: bool = False,
                 residual_scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.use_residual = use_residual
        self.residual_scale = residual_scale

        # Simple 1x1 conv to fuse multi-scale features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        """Forward pass.

        Args:
            inputs (list[Tensor]): Multi-scale backbone outputs.

        Returns:
            Tensor: Segmentation logits (B, num_classes, H, W).
        """
        x = self._transform_inputs(inputs)
        x = self.bottleneck(x)
        output = self.cls_seg(x)
        return output
