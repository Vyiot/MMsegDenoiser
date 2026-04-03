"""DenoiserSegmentor: an EncoderDecoder wrapper for pseudo-label denoising.

This module wraps mmseg's EncoderDecoder to handle the augmented input
(satellite image + one-hot pseudo-label concatenated along channel dim).
The backbone's first convolution is adapted to accept (3 + num_classes)
input channels.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Optional

from mmcv.runner import auto_fp16
from mmseg.models.builder import SEGMENTORS, build_backbone, build_head, build_neck
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize


@SEGMENTORS.register_module()
class DenoiserSegmentor(BaseSegmentor):
    """Pseudo-label denoiser built on top of mmseg's segmentor interface.

    The model takes as input the concatenation of:
        - Satellite image: (B, 3, H, W)
        - One-hot pseudo-label: (B, num_classes, H, W)
    producing a combined input of shape (B, 3 + num_classes, H, W).

    The backbone's stem convolution is automatically adapted to accept
    the expanded channel count.

    Args:
        backbone (dict): Backbone config.
        decode_head (dict): Decode head config.
        neck (dict, optional): Neck config.
        auxiliary_head (dict, optional): Auxiliary head config.
        num_classes (int): Number of segmentation classes.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        pretrained (str, optional): Pretrained model path for backbone.
        init_cfg (dict, optional): Initialization config.
        adapt_input_conv (bool): Whether to adapt the first conv layer
            of the backbone to accept (3 + num_classes) channels.
            Default: True.
    """

    def __init__(self,
                 backbone: dict,
                 decode_head: dict,
                 neck: Optional[dict] = None,
                 auxiliary_head: Optional[dict] = None,
                 num_classes: int = 7,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[dict] = None,
                 adapt_input_conv: bool = True):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = 3 + num_classes  # RGB + one-hot pseudo-label

        # Build backbone
        if pretrained is not None:
            backbone['pretrained'] = pretrained
        self.backbone = build_backbone(backbone)

        # Adapt the first convolution to accept expanded input channels
        if adapt_input_conv:
            self._adapt_first_conv(self.in_channels)

        # Build neck
        if neck is not None:
            self.neck = build_neck(neck)

        # Build decode head
        self._init_decode_head(decode_head)

        # Build auxiliary head
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _adapt_first_conv(self, new_in_channels: int):
        """Adapt the backbone's first convolution to accept more channels.

        Strategy: keep the pretrained weights for the first 3 channels
        and initialize the remaining channels with a scaled Kaiming init.
        """
        # Find the first Conv2d in the backbone
        first_conv = None
        first_conv_name = None

        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                break

        if first_conv is None or first_conv.in_channels == new_in_channels:
            return

        old_in_channels = first_conv.in_channels
        new_conv = nn.Conv2d(
            new_in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
        )

        # Copy pretrained weights for the first `old_in_channels` channels
        with torch.no_grad():
            new_conv.weight[:, :old_in_channels] = first_conv.weight
            # Initialize extra channels: scale by sqrt(old/new) for variance
            nn.init.kaiming_normal_(
                new_conv.weight[:, old_in_channels:], mode='fan_out',
                nonlinearity='relu')
            new_conv.weight[:, old_in_channels:] *= (
                old_in_channels / new_in_channels) ** 0.5
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Replace the conv in the backbone
        parts = first_conv_name.split('.')
        parent = self.backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    def _init_decode_head(self, decode_head: dict):
        """Initialize decode head."""
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: Optional[dict]):
        """Initialize auxiliary head."""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from the combined input.

        Args:
            img (Tensor): Combined input (B, 3+num_classes, H, W).

        Returns:
            list[Tensor]: Multi-scale backbone feature maps.
        """
        x = self.backbone(img)
        if hasattr(self, 'neck') and self.neck is not None:
            x = self.neck(x)
        return x

    def encode_decode(self, img: torch.Tensor,
                      img_metas: List[dict]) -> torch.Tensor:
        """Encode-decode for inference.

        Args:
            img (Tensor): Combined input (B, 3+num_classes, H, W).
            img_metas (list[dict]): Image meta information.

        Returns:
            Tensor: Segmentation logits.
        """
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run decode head forward in training mode."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(loss_decode)
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run decode head forward in testing mode."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run auxiliary head forward in training mode."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update({f'aux_{idx}.{k}': v for k, v in loss_aux.items()})
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update({f'aux.{k}': v for k, v in loss_aux.items()})
        return losses

    @auto_fp16(apply_to=('img',))
    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward pass for training.

        Args:
            img (Tensor): Combined input (B, 3+num_classes, H, W).
            img_metas (list[dict]): Image meta information.
            gt_semantic_seg (Tensor): Clean ground-truth labels (B, 1, H, W).

        Returns:
            dict[str, Tensor]: Loss dictionary.
        """
        x = self.extract_feat(img)
        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """Forward pass for testing (single-scale)."""
        img = imgs[0]
        img_meta = img_metas[0]
        if not isinstance(img_meta, list):
            img_meta = [img_meta]
        return self.simple_test(img, img_meta, **kwargs)

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        """Simple test with single scale."""
        if img.dim() == 3:
            img = img.unsqueeze(0)
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                input=seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # Unpack batch
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        """Multi-scale augmented test (placeholder)."""
        img = imgs[0]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        return self.simple_test(img, img_metas[0], rescale, **kwargs)
