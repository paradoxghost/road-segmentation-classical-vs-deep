"""Loss functions for binary road segmentation."""

from __future__ import annotations

import torch
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute soft Dice loss from logits and binary targets."""
    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = torch.sum(probabilities * targets, dim=1)
    denominator = torch.sum(probabilities, dim=1) + torch.sum(targets, dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)

    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combine BCEWithLogitsLoss with Dice loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)
