"""FusionNeck: multi-scale feature fusion for dual-encoder architectures.

Merges features from two separate backbones (RGB and pseudo-label) at
each spatial scale. Supports multiple fusion strategies:

    - 'concat':    Channel-wise concatenation followed by 1x1 conv reduction.
    - 'add':       Element-wise addition (requires matching channel dims).
    - 'crossattn': Cross-attention where RGB features attend to PL features
                   and vice versa, then summed.

Each strategy is applied independently at every feature scale, allowing
the network to learn scale-specific cross-modal interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from mmcv.runner import BaseModule
from mmseg.models.builder import NECKS


@NECKS.register_module()
class FusionNeck(BaseModule):
    """Multi-scale feature fusion neck for dual-encoder segmentors.

    Args:
        in_channels_rgb (list[int]): Channel dims at each scale from RGB backbone.
            E.g., [64, 128, 320, 512] for MiT-B2, [256, 512, 1024, 2048] for ResNet.
        in_channels_pl (list[int]): Channel dims at each scale from PL backbone.
            Should have the same number of scales as in_channels_rgb.
        out_channels (list[int], optional): Output channel dims at each scale.
            If None, defaults to in_channels_rgb.
        fusion_type (str): Fusion strategy. One of 'concat', 'add', 'crossattn'.
            Default: 'concat'.
        norm_cfg (dict): Normalization config. Default: dict(type='BN').
        act_cfg (dict): Activation config. Default: dict(type='ReLU').
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 in_channels_rgb: List[int],
                 in_channels_pl: List[int],
                 out_channels: List[int] = None,
                 fusion_type: str = 'concat',
                 norm_cfg: dict = dict(type='BN'),
                 act_cfg: dict = dict(type='ReLU'),
                 init_cfg: dict = None):
        super().__init__(init_cfg=init_cfg)
        assert len(in_channels_rgb) == len(in_channels_pl), \
            'RGB and PL backbones must produce the same number of feature scales.'
        assert fusion_type in ('concat', 'add', 'crossattn'), \
            f'Unsupported fusion_type: {fusion_type}'

        self.num_scales = len(in_channels_rgb)
        self.fusion_type = fusion_type

        if out_channels is None:
            out_channels = list(in_channels_rgb)
        self.out_channels = out_channels

        # Build per-scale fusion modules
        if fusion_type == 'concat':
            self._build_concat_fusion(in_channels_rgb, in_channels_pl,
                                      out_channels, norm_cfg)
        elif fusion_type == 'add':
            self._build_add_fusion(in_channels_rgb, in_channels_pl,
                                   out_channels, norm_cfg)
        elif fusion_type == 'crossattn':
            self._build_crossattn_fusion(in_channels_rgb, in_channels_pl,
                                         out_channels, norm_cfg)

    def _build_concat_fusion(self, ch_rgb, ch_pl, ch_out, norm_cfg):
        """Concatenate + 1x1 conv reduction at each scale."""
        self.fuse_convs = nn.ModuleList()
        for i in range(self.num_scales):
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(ch_rgb[i] + ch_pl[i], ch_out[i], 1, bias=False),
                nn.BatchNorm2d(ch_out[i]),
                nn.ReLU(inplace=True),
            ))

    def _build_add_fusion(self, ch_rgb, ch_pl, ch_out, norm_cfg):
        """Project PL features to match RGB dims, then element-wise add."""
        self.proj_pl = nn.ModuleList()
        self.proj_rgb = nn.ModuleList()
        for i in range(self.num_scales):
            # Project both to out_channels
            self.proj_rgb.append(
                nn.Sequential(
                    nn.Conv2d(ch_rgb[i], ch_out[i], 1, bias=False),
                    nn.BatchNorm2d(ch_out[i]),
                ) if ch_rgb[i] != ch_out[i] else nn.Identity()
            )
            self.proj_pl.append(
                nn.Sequential(
                    nn.Conv2d(ch_pl[i], ch_out[i], 1, bias=False),
                    nn.BatchNorm2d(ch_out[i]),
                ) if ch_pl[i] != ch_out[i] else nn.Identity()
            )
        self.relu = nn.ReLU(inplace=True)

    def _build_crossattn_fusion(self, ch_rgb, ch_pl, ch_out, norm_cfg):
        """Lightweight cross-attention fusion at each scale.

        For each scale, RGB features query PL features (and vice versa)
        via a single-head spatial cross-attention with reduced dim.
        """
        self.attn_modules = nn.ModuleList()
        for i in range(self.num_scales):
            embed_dim = min(ch_rgb[i], ch_pl[i], 64)  # compact attention
            self.attn_modules.append(
                CrossModalAttention(
                    dim_rgb=ch_rgb[i],
                    dim_pl=ch_pl[i],
                    embed_dim=embed_dim,
                    out_dim=ch_out[i]))

    def forward(self, feats_rgb: Tuple[torch.Tensor, ...],
                feats_pl: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        """Fuse multi-scale features from two backbones.

        Args:
            feats_rgb (tuple[Tensor]): Multi-scale features from RGB backbone.
                Each tensor has shape (B, C_i, H_i, W_i).
            feats_pl (tuple[Tensor]): Multi-scale features from PL backbone.

        Returns:
            list[Tensor]: Fused features at each scale.
        """
        assert len(feats_rgb) == len(feats_pl) == self.num_scales

        fused = []
        for i in range(self.num_scales):
            f_rgb = feats_rgb[i]
            f_pl = feats_pl[i]

            # Handle spatial size mismatch (e.g., different strides)
            if f_rgb.shape[2:] != f_pl.shape[2:]:
                f_pl = F.interpolate(
                    f_pl, size=f_rgb.shape[2:],
                    mode='bilinear', align_corners=False)

            if self.fusion_type == 'concat':
                out = self.fuse_convs[i](torch.cat([f_rgb, f_pl], dim=1))
            elif self.fusion_type == 'add':
                out = self.relu(self.proj_rgb[i](f_rgb) +
                                self.proj_pl[i](f_pl))
            elif self.fusion_type == 'crossattn':
                out = self.attn_modules[i](f_rgb, f_pl)

            fused.append(out)

        return fused


class CrossModalAttention(nn.Module):
    """Lightweight cross-modal spatial attention.

    RGB features attend to PL features to capture semantic guidance,
    and PL features attend to RGB features to capture appearance cues.
    Results are summed and projected to the output dimension.

    Args:
        dim_rgb (int): Channel dim of RGB features.
        dim_pl (int): Channel dim of PL features.
        embed_dim (int): Internal attention dimension.
        out_dim (int): Output channel dimension.
    """

    def __init__(self, dim_rgb: int, dim_pl: int,
                 embed_dim: int, out_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # RGB queries PL
        self.q_rgb = nn.Conv2d(dim_rgb, embed_dim, 1)
        self.k_pl = nn.Conv2d(dim_pl, embed_dim, 1)
        self.v_pl = nn.Conv2d(dim_pl, embed_dim, 1)

        # PL queries RGB
        self.q_pl = nn.Conv2d(dim_pl, embed_dim, 1)
        self.k_rgb = nn.Conv2d(dim_rgb, embed_dim, 1)
        self.v_rgb = nn.Conv2d(dim_rgb, embed_dim, 1)

        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim * 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.scale = embed_dim ** -0.5

    def _spatial_attention(self, q, k, v):
        """Compute spatial cross-attention.

        Args:
            q (Tensor): (B, D, H, W) query features.
            k (Tensor): (B, D, H, W) key features.
            v (Tensor): (B, D, H, W) value features.

        Returns:
            Tensor: (B, D, H, W) attended features.
        """
        B, D, H, W = q.shape
        N = H * W

        q = q.view(B, D, N).permute(0, 2, 1)  # (B, N, D)
        k = k.view(B, D, N)                     # (B, D, N)
        v = v.view(B, D, N).permute(0, 2, 1)   # (B, N, D)

        attn = torch.bmm(q, k) * self.scale     # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                 # (B, N, D)
        out = out.permute(0, 2, 1).view(B, D, H, W)
        return out

    def forward(self, f_rgb: torch.Tensor,
                f_pl: torch.Tensor) -> torch.Tensor:
        # RGB attends to PL (semantic guidance)
        out_rgb2pl = self._spatial_attention(
            self.q_rgb(f_rgb), self.k_pl(f_pl), self.v_pl(f_pl))

        # PL attends to RGB (appearance cues)
        out_pl2rgb = self._spatial_attention(
            self.q_pl(f_pl), self.k_rgb(f_rgb), self.v_rgb(f_rgb))

        # Combine both directions
        combined = torch.cat([out_rgb2pl, out_pl2rgb], dim=1)
        return self.proj(combined)
