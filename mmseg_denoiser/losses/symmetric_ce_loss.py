"""Symmetric Cross-Entropy Loss for learning with noisy labels.

Reference:
    Wang et al., "Symmetric Cross Entropy for Robust Learning with Noisy
    Labels", ICCV 2019.

The symmetric CE loss combines the standard CE loss with a reverse CE term:
    L_sce = α * CE(p, q) + β * CE(q, p)
where p is the predicted distribution and q is the (potentially noisy) label.
The reverse term acts as a regularizer that is more robust to label noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES
from mmseg.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class SymmetricCrossEntropyLoss(nn.Module):
    """Symmetric Cross-Entropy Loss.

    Args:
        alpha (float): Weight for the standard CE term. Default: 1.0.
        beta (float): Weight for the reverse CE term. Default: 1.0.
        num_classes (int): Number of classes. Default: 7.
        ignore_index (int): Index to ignore. Default: 255.
        loss_weight (float): Global weight for this loss. Default: 1.0.
        clamp_value (float): Clamp value to avoid log(0). Default: 1e-7.
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 num_classes: int = 7,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0,
                 clamp_value: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.clamp_value = clamp_value

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
            weight (Tensor, optional): Per-sample weights.
            avg_factor (int, optional): Average factor for loss.
            reduction_override (str, optional): Reduction method.

        Returns:
            Tensor: Computed loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else 'mean'

        # Standard CE term: -q * log(p)
        ce_loss = F.cross_entropy(
            pred, target,
            ignore_index=self.ignore_index,
            reduction='none')

        # Reverse CE term: -p * log(q)
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=self.clamp_value,
                                   max=1.0 - self.clamp_value)

        # One-hot encode target
        valid_mask = (target != self.ignore_index).float()
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_onehot = F.one_hot(
            target_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        target_onehot = torch.clamp(target_onehot, min=self.clamp_value,
                                    max=1.0 - self.clamp_value)

        # Reverse CE: -sum_c p_c * log(q_c)
        rce_loss = -(pred_softmax * torch.log(target_onehot)).sum(dim=1)
        rce_loss = rce_loss * valid_mask

        # Combine
        loss = self.alpha * ce_loss + self.beta * rce_loss

        # Apply weight and reduction
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

        return self.loss_weight * loss
