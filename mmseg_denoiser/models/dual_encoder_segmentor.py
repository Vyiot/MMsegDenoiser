"""DualEncoderSegmentor: late fusion architecture for pseudo-label denoising.

Unlike the early fusion DenoiserSegmentor which concatenates the satellite
image and pseudo-label at the input level, this architecture processes
each modality through a separate encoder backbone and fuses features at
multiple scales via a FusionNeck before the decode head.

Architecture:
    Satellite Image (B, 3, H, W)  --> [backbone_rgb]  --> {F_rgb_i}
                                                                 \
                                                          [FusionNeck] --> Decode Head --> Output
                                                                 /
    Pseudo-label    (B, C, H, W)  --> [backbone_pl]   --> {F_pl_i}

The RGB backbone is initialized from ImageNet pretrained weights and its
first conv layer remains intact (3 input channels). The pseudo-label
backbone has its first conv adapted to accept num_classes channels.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from mmcv.runner import auto_fp16
from mmseg.models.builder import (SEGMENTORS, build_backbone, build_head,
                                  build_neck)
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize


@SEGMENTORS.register_module()
class DualEncoderSegmentor(BaseSegmentor):
    """Dual-encoder (late fusion) segmentor for pseudo-label denoising.

    Two separate backbones process the satellite image and the one-hot
    pseudo-label independently. Their multi-scale features are fused
    via a FusionNeck before being passed to the decode head.

    Args:
        backbone_rgb (dict): Config for the satellite image backbone.
        backbone_pl (dict): Config for the pseudo-label backbone.
            If None, a lightweight version of backbone_rgb is used.
        fusion_neck (dict): Config for the FusionNeck that merges features.
        decode_head (dict): Config for the decode head.
        auxiliary_head (dict, optional): Config for auxiliary head.
        num_classes (int): Number of segmentation classes.
        train_cfg (dict, optional): Training config.
        test_cfg (dict, optional): Testing config.
        pretrained_rgb (str, optional): Pretrained weights for RGB backbone.
        pretrained_pl (str, optional): Pretrained weights for PL backbone.
            If provided, first conv is adapted; extra channels use Kaiming init.
        init_cfg (dict, optional): Initialization config.
        share_backbone (bool): If True, share weights between the two
            backbones (with separate first conv). Default: False.
    """

    def __init__(self,
                 backbone_rgb: dict,
                 backbone_pl: dict,
                 fusion_neck: dict,
                 decode_head: dict,
                 auxiliary_head: Optional[dict] = None,
                 num_classes: int = 7,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 pretrained_rgb: Optional[str] = None,
                 pretrained_pl: Optional[str] = None,
                 init_cfg: Optional[dict] = None,
                 share_backbone: bool = False):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes

        # ---- RGB backbone (standard 3-channel input) ----
        if pretrained_rgb is not None:
            backbone_rgb['pretrained'] = pretrained_rgb
        self.backbone_rgb = build_backbone(backbone_rgb)

        # ---- Pseudo-label backbone (num_classes-channel input) ----
        if pretrained_pl is not None:
            backbone_pl['pretrained'] = pretrained_pl
        self.backbone_pl = build_backbone(backbone_pl)
        self._adapt_first_conv(self.backbone_pl, num_classes)

        # Optionally share weights (except first conv)
        if share_backbone:
            self._share_backbone_weights()

        # ---- Fusion neck ----
        self.fusion_neck = build_neck(fusion_neck)

        # ---- Decode head ----
        self._init_decode_head(decode_head)

        # ---- Auxiliary head ----
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @staticmethod
    def _adapt_first_conv(backbone: nn.Module, new_in_channels: int):
        """Adapt the first Conv2d of a backbone to accept new_in_channels."""
        first_conv = None
        first_conv_name = None
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                first_conv_name = name
                break

        if first_conv is None or first_conv.in_channels == new_in_channels:
            return

        old_in = first_conv.in_channels
        new_conv = nn.Conv2d(
            new_in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None)

        with torch.no_grad():
            # If pretrained, reuse weights for overlapping channels
            copy_channels = min(old_in, new_in_channels)
            new_conv.weight[:, :copy_channels] = first_conv.weight[:, :copy_channels]
            if new_in_channels > old_in:
                nn.init.kaiming_normal_(
                    new_conv.weight[:, old_in:],
                    mode='fan_out', nonlinearity='relu')
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        parts = first_conv_name.split('.')
        parent = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    def _share_backbone_weights(self):
        """Share all backbone parameters except the first conv layer."""
        # Identify first conv in each backbone
        rgb_first = None
        pl_first = None
        for name, _ in self.backbone_rgb.named_modules():
            if isinstance(_, nn.Conv2d):
                rgb_first = name
                break
        for name, _ in self.backbone_pl.named_modules():
            if isinstance(_, nn.Conv2d):
                pl_first = name
                break

        for (name_rgb, param_rgb), (name_pl, param_pl) in zip(
                self.backbone_rgb.named_parameters(),
                self.backbone_pl.named_parameters()):
            # Skip first conv (different input channels)
            if rgb_first and name_rgb.startswith(rgb_first):
                continue
            param_pl.data = param_rgb.data

    def _init_decode_head(self, decode_head: dict):
        self.decode_head = build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head: Optional[dict]):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList(
                    [build_head(h) for h in auxiliary_head])
            else:
                self.auxiliary_head = build_head(auxiliary_head)

    def extract_feat(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Extract and fuse features from both modalities.

        Args:
            img (Tensor): Combined input (B, 3 + num_classes, H, W).
                The first 3 channels are the satellite image, the
                remaining num_classes channels are the one-hot pseudo-label.

        Returns:
            list[Tensor]: Fused multi-scale features from FusionNeck.
        """
        # Split input along channel dimension
        img_rgb = img[:, :3, :, :]
        img_pl = img[:, 3:, :, :]

        # Forward through separate backbones
        feats_rgb = self.backbone_rgb(img_rgb)
        feats_pl = self.backbone_pl(img_pl)

        # Fuse multi-scale features
        fused = self.fusion_neck(feats_rgb, feats_pl)
        return fused

    def encode_decode(self, img: torch.Tensor,
                      img_metas: List[dict]) -> torch.Tensor:
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        loss_decode = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(loss_decode)
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        return self.decode_head.forward_test(x, img_metas, self.test_cfg)

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(
                    x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update({f'aux_{idx}.{k}': v
                               for k, v in loss_aux.items()})
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update({f'aux.{k}': v for k, v in loss_aux.items()})
        return losses

    @auto_fp16(apply_to=('img',))
    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
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
        img_meta = img_metas[0]
        if not isinstance(img_meta, list):
            img_meta = [img_meta]
        return self.simple_test(imgs[0], img_meta, **kwargs)

    def simple_test(self, img, img_meta, rescale=True, **kwargs):
        if img.dim() == 3:
            img = img.unsqueeze(0)
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # Handle list of dicts consistent with mmseg/DenoiserSegmentor
            if isinstance(img_meta, list):
                size = img_meta[0]['ori_shape'][:2]
            else:
                size = img_meta['ori_shape'][:2]
            seg_logit = resize(
                input=seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        return list(seg_pred)

    def aug_test(self, imgs, img_metas, rescale=True, **kwargs):
        return self.simple_test(imgs[0], img_metas[0], rescale, **kwargs)
