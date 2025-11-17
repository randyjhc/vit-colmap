"""
Loss functions for training ViT-based feature extractors.
"""

from .feature_losses import (
    DetectorLoss,
    RotationEquivarianceLoss,
    DescriptorLoss,
    TotalLoss,
)

__all__ = [
    "DetectorLoss",
    "RotationEquivarianceLoss",
    "DescriptorLoss",
    "TotalLoss",
]
