"""Noise-Robust Dice Loss for segmentation with noisy labels.

Dice loss is inherently more robust to class imbalance and can complement
CE-based losses in the noisy label setting. This implementation adds
label smoothing to further mitigate the impact of noisy annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class NoiseRobustDiceLoss(nn.Module):
    """Dice loss with label smoothing for noisy-label robustness.

    Args:
        smooth (float): Smoothing factor for Dice denominator. Default: 1.0.
        label_smoothing (float): Label smoothing factor [0, 1). Default: 0.1.
        num_classes (int): Number of classes. Default: 7.
        ignore_index (int): Index to ignore. Default: 255.
        loss_weight (float): Weight of this loss. Default: 1.0.
        exponent (int): Exponent in Dice denominator. Default: 2.
    """

    def __init__(self,
                 smooth: float = 1.0,
                 label_smoothing: float = 0.1,
                 num_classes: int = 7,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0,
                 exponent: int = 2):
        super().__init__()
        self.smooth = smooth
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.exponent = exponent

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: torch.Tensor = None,
                avg_factor: int = None,
                reduction_override: str = None,
                **kwargs) -> torch.Tensor:
        """Forward computation.

        Args:
            pred (Tensor): Prediction logits (B, C, H, W).
            target (Tensor): Ground-truth labels (B, H, W).

        Returns:
            Tensor: Dice loss value.
        """
        pred_softmax = F.softmax(pred, dim=1)  # (B, C, H, W)

        # Create valid mask
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)

        # One-hot encode target with label smoothing
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_onehot = F.one_hot(
            target_clamped, self.num_classes).permute(0, 3, 1, 2).float()

        # Apply label smoothing: (1 - ε) * one_hot + ε / C
        if self.label_smoothing > 0:
            target_onehot = (
                (1.0 - self.label_smoothing) * target_onehot +
                self.label_smoothing / self.num_classes)

        # Apply valid mask
        pred_softmax = pred_softmax * valid_mask
        target_onehot = target_onehot * valid_mask

        # Per-class Dice
        dims = (0, 2, 3)  # Reduce over batch and spatial dims
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        cardinality = (pred_softmax.pow(self.exponent) +
                       target_onehot.pow(self.exponent)).sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        # Average over classes
        loss = dice_loss.mean()

        return self.loss_weight * loss
