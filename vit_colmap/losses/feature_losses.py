"""
Loss functions for self-supervised feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DetectorLoss(nn.Module):
    """Binary cross-entropy loss for keypoint detection."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        score_logits: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute BCE loss for keypoint detection.

        Args:
            score_logits: (B, 1, H, W) or (B, H, W) raw score logits
            targets: (B, 1, H, W) or (B, H, W) ground truth heatmap [0, 1]
            weights: Optional (B, 1, H, W) or (B, H, W) importance weights

        Returns:
            Scalar loss value
        """
        # Ensure consistent shapes
        if score_logits.dim() == 3:
            score_logits = score_logits.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if weights is not None and weights.dim() == 3:
            weights = weights.unsqueeze(1)

        loss = F.binary_cross_entropy_with_logits(
            score_logits, targets, weight=weights, reduction="mean"
        )
        return loss


class RotationEquivarianceLoss(nn.Module):
    """Loss for rotation equivariance of keypoint orientations."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        orientations1: torch.Tensor,
        orientations2: torch.Tensor,
        rotation_angle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure orientation2 ≈ orientation1 + rotation_angle.

        Uses circular loss to handle wraparound at ±π.

        Args:
            orientations1: (B, K) orientations in image 1 (radians)
            orientations2: (B, K) orientations in image 2 (radians)
            rotation_angle: (B,) or (B, K) rotation from image 1 to 2 (radians)

        Returns:
            Scalar loss value
        """
        # Expand rotation_angle if needed
        if rotation_angle.dim() == 1:
            rotation_angle = rotation_angle.unsqueeze(1)  # (B, 1)

        # Expected orientation in image 2
        expected_orientation2 = orientations1 + rotation_angle

        # Circular difference (handles wraparound)
        diff = torch.atan2(
            torch.sin(orientations2 - expected_orientation2),
            torch.cos(orientations2 - expected_orientation2),
        )

        # L2 loss on angular difference
        loss = diff.pow(2).mean()
        return loss


class DescriptorLoss(nn.Module):
    """Contrastive loss for descriptor learning."""

    def __init__(self, margin: float = 0.5, use_hard_negative: bool = True):
        """
        Args:
            margin: Margin for triplet loss
            use_hard_negative: If True, use hardest negative; else use all negatives
        """
        super().__init__()
        self.margin = margin
        self.use_hard_negative = use_hard_negative

    def positive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Positive pair loss: L_pos = 1 - <z1, z2>.

        Args:
            z1: (B, K, D) L2-normalized descriptors from image 1
            z2: (B, K, D) L2-normalized descriptors from image 2

        Returns:
            (B, K) loss per point
        """
        # Cosine similarity (z1 and z2 should already be normalized)
        similarity = (z1 * z2).sum(dim=-1)  # (B, K)
        loss = 1.0 - similarity
        return loss

    def triplet_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Triplet loss: L = max(0, m + d(anchor, positive) - d(anchor, negative)).

        Args:
            anchor: (B, K, D) anchor descriptors
            positive: (B, K, D) positive descriptors
            negative: (B, K, N, D) negative descriptors

        Returns:
            (B, K) loss per point
        """
        # Distance to positive (1 - cosine similarity)
        d_pos = 1.0 - (anchor * positive).sum(dim=-1)  # (B, K)

        # Distance to negatives
        # anchor: (B, K, D), negative: (B, K, N, D)
        # Compute: anchor @ negative^T for each (b, k)
        d_neg = 1.0 - torch.einsum("bkd,bknd->bkn", anchor, negative)  # (B, K, N)

        if self.use_hard_negative:
            # Use hardest negative (smallest distance)
            d_neg_selected = d_neg.min(dim=-1)[0]  # (B, K)
        else:
            # Use mean distance to all negatives
            d_neg_selected = d_neg.mean(dim=-1)  # (B, K)

        # Triplet loss
        loss = F.relu(self.margin + d_pos - d_neg_selected)
        return loss

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        negatives: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combined positive and triplet loss.

        Args:
            z1: (B, K, D) descriptors from image 1
            z2: (B, K, D) descriptors from image 2
            negatives: (B, K, N, D) negative descriptors
            weights: Optional (B, K) weights for each point (e.g., keypoint scores)

        Returns:
            Scalar loss value
        """
        L_pos = self.positive_loss(z1, z2)  # (B, K)
        L_triplet = self.triplet_loss(z1, z2, negatives)  # (B, K)

        # Combined loss per point
        L_combined = L_pos + L_triplet  # (B, K)

        if weights is not None:
            # Weighted mean
            loss = (weights * L_combined).sum() / (weights.sum() + 1e-8)
        else:
            loss = L_combined.mean()

        return loss


class TotalLoss(nn.Module):
    """Aggregated loss for keypoint detection and descriptor learning."""

    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_rot: float = 0.5,
        lambda_desc: float = 1.0,
        margin: float = 0.5,
    ):
        """
        Args:
            lambda_det: Weight for detector loss
            lambda_rot: Weight for rotation equivariance loss
            lambda_desc: Weight for descriptor loss
            margin: Margin for triplet loss
        """
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_rot = lambda_rot
        self.lambda_desc = lambda_desc

        self.detector_loss = DetectorLoss()
        self.rotation_loss = RotationEquivarianceLoss()
        self.descriptor_loss = DescriptorLoss(margin=margin)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with keypoint score weighting.

        Args:
            outputs: Dict containing:
                - score_logits: (B, 1, H, W) keypoint score logits
                - orientations1: (B, K) orientations in image 1
                - orientations2: (B, K) orientations in image 2
                - z1: (B, K, 128) descriptors from image 1
                - z2: (B, K, 128) descriptors from image 2
                - negatives: (B, K, N, 128) negative descriptors

            targets: Dict containing:
                - score_gt: (B, 1, H, W) ground truth score heatmap
                - rotation_angle: (B,) rotation angle from image 1 to 2
                - invariant_coords: (B, K, 2) coordinates of invariant points

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # 1. Keypoint detection loss (BCE)
        if "score_logits" in outputs and "score_gt" in targets:
            L_det = self.detector_loss(outputs["score_logits"], targets["score_gt"])
            losses["detector"] = L_det
        else:
            L_det = torch.tensor(0.0, device=outputs["z1"].device)
            losses["detector"] = L_det

        # 2. Rotation equivariance loss
        if "rotation_angle" in targets:
            L_rot = self.rotation_loss(
                outputs["orientations1"],
                outputs["orientations2"],
                targets["rotation_angle"],
            )
            losses["rotation"] = L_rot
        else:
            L_rot = torch.tensor(0.0, device=outputs["z1"].device)
            losses["rotation"] = L_rot

        # 3. Descriptor loss weighted by keypoint scores
        # w(p) = sigmoid(S_logit(p))
        if "score_logits" in outputs and "invariant_coords" in targets:
            # Sample score logits at invariant point locations
            score_weights = self._sample_scores_at_coords(
                outputs["score_logits"], targets["invariant_coords"]
            )
            score_weights = torch.sigmoid(score_weights)  # (B, K)
        else:
            score_weights = None

        L_desc = self.descriptor_loss(
            outputs["z1"],
            outputs["z2"],
            outputs["negatives"],
            weights=score_weights,
        )
        losses["descriptor"] = L_desc

        # 4. Total loss
        L_total = (
            self.lambda_det * L_det
            + self.lambda_rot * L_rot
            + self.lambda_desc * L_desc
        )
        losses["total"] = L_total

        return losses

    def _sample_scores_at_coords(
        self,
        score_logits: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample score logits at specified coordinates.

        Args:
            score_logits: (B, 1, H, W) score map
            coords: (B, K, 2) coordinates (x, y) in feature map space

        Returns:
            scores: (B, K) sampled scores
        """
        B, _, H, W = score_logits.shape

        # Normalize coordinates to [-1, 1] for grid_sample
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # x
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # y

        # Reshape for grid_sample: (B, K, 1, 2)
        grid = grid.unsqueeze(2)

        # Sample: output is (B, 1, K, 1)
        sampled = F.grid_sample(
            score_logits, grid, mode="bilinear", align_corners=False
        )

        # Reshape to (B, K)
        scores = sampled.squeeze(1).squeeze(-1)
        return scores
