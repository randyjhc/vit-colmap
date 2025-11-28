"""
Loss functions for training ViT-based feature extractors.
"""

from .feature_losses import (
    DetectorLoss,
    DescriptorLoss,
    TotalLoss,
)

__all__ = [
    "DetectorLoss",
    "DescriptorLoss",
    "TotalLoss",
]
