"""
Trainable ViT-based feature extraction model.
Uses DINOv2 as backbone with learnable upsampling and detection heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class UpsampleBlock(nn.Module):
    """Single upsampling block: ConvTranspose2d + Conv3x3 + BatchNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ViTFeatureModel(nn.Module):
    """
    Trainable model for keypoint and descriptor extraction.

    Architecture:
    1. Frozen DINOv2 backbone (768D or 1024D features at 1/14 resolution)
    2. Progressive upsampling to 1/2 original resolution
    3. Shared trunk for feature refinement
    4. Two heads: keypoint detection (4D) and descriptor extraction (128D)
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        descriptor_dim: int = 128,
        freeze_backbone: bool = True,
    ):
        """
        Initialize the model.

        Args:
            backbone_name: DINOv2 model name ('dinov2_vitb14', 'dinov2_vitl14', etc.)
            descriptor_dim: Output descriptor dimension (default: 128)
            freeze_backbone: Whether to freeze DINOv2 weights (default: True)
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.descriptor_dim = descriptor_dim
        self.patch_size = 14  # DINOv2 uses 14x14 patches

        # Load DINOv2 backbone
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )

        # Determine backbone feature dimension
        if "vitb" in backbone_name:
            self.backbone_dim = 768
        elif "vitl" in backbone_name:
            self.backbone_dim = 1024
        elif "vitg" in backbone_name:
            self.backbone_dim = 1536
        else:
            # Try to infer from model
            self.backbone_dim = self.backbone.embed_dim

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Build upsampler: from 1/14 to 1/2 resolution
        # Need ~7x upsampling (14/2 = 7)
        # Using 3 upsample blocks: 2^3 = 8x (slightly more than needed)
        # Actual: 1/14 -> 1/7 -> 1/3.5 -> 1/1.75 (we'll adjust final size)
        self.upsampler = nn.Sequential(
            # Block 1: backbone_dim -> 512, 2x upsample
            UpsampleBlock(self.backbone_dim, 512),
            # Block 2: 512 -> 512, 2x upsample
            UpsampleBlock(512, 512),
            # Block 3: 512 -> 512, 2x upsample
            UpsampleBlock(512, 512),
        )

        # Shared trunk: 512 -> 256
        self.trunk = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Keypoint head: predicts (score, x_offset, y_offset, orientation)
        # score: confidence that this pixel is a keypoint
        # x_offset, y_offset: sub-pixel refinement
        # orientation: keypoint orientation in radians
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 4, kernel_size=1),  # 4 channels: score, dx, dy, orientation
        )

        # Descriptor head: 256 -> 128D descriptors
        self.descriptor_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, descriptor_dim, kernel_size=1),
        )

        print("ViTFeatureModel initialized:")
        print(f"  Backbone: {backbone_name} ({self.backbone_dim}D)")
        print(f"  Descriptor dim: {descriptor_dim}")
        print(f"  Backbone frozen: {freeze_backbone}")

    def _extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from DINOv2 backbone.

        Args:
            x: Input image tensor (B, 3, H, W)
               H and W must be divisible by patch_size (14)

        Returns:
            Feature tensor (B, C, H/14, W/14)
        """
        B, C, H, W = x.shape

        # Ensure dimensions are compatible
        assert (
            H % self.patch_size == 0
        ), f"Height {H} not divisible by patch_size {self.patch_size}"
        assert (
            W % self.patch_size == 0
        ), f"Width {W} not divisible by patch_size {self.patch_size}"

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # Extract features from backbone
        if self.backbone.training:
            features = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                features = self.backbone.forward_features(x)

        # Handle different DINOv2 output formats
        if isinstance(features, dict):
            patch_features = features["x_norm_patchtokens"]
        else:
            # Skip CLS token at position 0
            patch_features = features[:, 1:, :]

        # Reshape to spatial grid: (B, N, C) -> (B, C, H, W)
        patch_features = patch_features.reshape(
            B, h_patches, w_patches, self.backbone_dim
        )
        patch_features = patch_features.permute(0, 3, 1, 2)  # (B, C, H, W)

        return patch_features

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W)
            target_size: Optional (H, W) to resize output feature maps to.
                        If None, outputs will be at ~1/2 resolution after upsampling.

        Returns:
            Dictionary containing:
                - 'keypoints': (B, 4, H_out, W_out) - score, dx, dy, orientation
                - 'descriptors': (B, 128, H_out, W_out) - dense descriptor map
                - 'features': (B, 256, H_out, W_out) - shared feature trunk output
        """
        B, C, H, W = x.shape

        # 1. Extract backbone features (1/14 resolution)
        backbone_features = self._extract_backbone_features(x)

        # 2. Upsample to higher resolution
        upsampled = self.upsampler(backbone_features)

        # 3. Adjust to target size (1/2 of input resolution)
        if target_size is None:
            target_h = H // 2
            target_w = W // 2
        else:
            target_h, target_w = target_size

        # Resize to exact target size using bilinear interpolation
        if upsampled.shape[2:] != (target_h, target_w):
            upsampled = F.interpolate(
                upsampled,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # 4. Shared trunk (B, 512, H/2, W/2) -> (B, 256, H/2, W/2)
        trunk_features = self.trunk(upsampled)

        # 5. Detection heads
        keypoints = self.keypoint_head(trunk_features)
        descriptors = self.descriptor_head(trunk_features)

        # Bound orientation to [-π, π] using tanh
        keypoints = keypoints.clone()
        keypoints[:, 3] = torch.tanh(keypoints[:, 3]) * torch.pi

        # Normalize descriptors to unit length
        descriptors = F.normalize(descriptors, p=2, dim=1)

        return {
            "keypoints": keypoints,  # (B, 4, H/2, W/2)
            "descriptors": descriptors,  # (B, 128, H/2, W/2)
            "features": trunk_features,  # (B, 256, H/2, W/2)
        }

    def get_trainable_parameters(self):
        """Get parameters that require gradients (excludes frozen backbone)."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }
