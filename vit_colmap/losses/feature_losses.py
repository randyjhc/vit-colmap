"""
Loss functions for self-supervised feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


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


class RepeatabilityLoss(nn.Module):
    """
    Encourages the model to detect keypoints at corresponding locations.
    This directly improves the number of potential matches.
    """

    def __init__(self, grid_size: int = 8):
        """
        Args:
            grid_size: Size of spatial grid for repeatability computation
        """
        super().__init__()
        self.grid_size = grid_size

    def forward(
        self,
        score_map1: torch.Tensor,
        score_map2: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute repeatability loss between two score maps.

        Args:
            score_map1: (B, 1, H, W) keypoint scores for image 1
            score_map2: (B, 1, H, W) keypoint scores for image 2
            homography: (B, 3, 3) homography from image 1 to image 2

        Returns:
            Scalar repeatability loss
        """
        B, _, H, W = score_map1.shape

        # Apply sigmoid to get probabilities
        prob1 = torch.sigmoid(score_map1)
        prob2 = torch.sigmoid(score_map2)

        # Warp prob1 to image 2 using homography
        prob1_warped = self._warp_with_homography(prob1, homography, (H, W))

        # L2 loss between warped probabilities
        loss = F.mse_loss(prob1_warped, prob2)

        return loss

    def _warp_with_homography(
        self,
        feature_map: torch.Tensor,
        homography: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Warp feature map using homography.

        Args:
            feature_map: (B, C, H, W) feature map
            homography: (B, 3, 3) homography matrix
            output_size: (H, W) output size

        Returns:
            Warped feature map (B, C, H, W)
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device

        # Create normalized coordinate grid [-1, 1]
        h_out, w_out = output_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, h_out, device=device),
            torch.linspace(-1, 1, w_out, device=device),
            indexing="ij",
        )

        # Convert to pixel coordinates [0, W-1], [0, H-1]
        x_grid_px = (x_grid + 1) * (w_out - 1) / 2
        y_grid_px = (y_grid + 1) * (h_out - 1) / 2

        # Homogeneous coordinates
        ones = torch.ones_like(x_grid_px)
        coords = torch.stack([x_grid_px, y_grid_px, ones], dim=-1)  # (H, W, 3)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

        # Apply homography: H @ [x, y, 1]^T
        coords_transformed = torch.einsum("bij,bhwj->bhwi", homography, coords)

        # Normalize by z coordinate
        z = coords_transformed[..., 2:3]
        coords_transformed = coords_transformed[..., :2] / (z + 1e-8)

        # Convert back to normalized coordinates [-1, 1]
        x_norm = 2 * coords_transformed[..., 0] / (W - 1) - 1
        y_norm = 2 * coords_transformed[..., 1] / (H - 1) - 1

        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B, H, W, 2)

        # Warp using grid_sample
        warped = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return warped


class MatchCountLoss(nn.Module):
    """
    Encourages more potential matches by maximizing descriptor similarity
    at corresponding locations while maintaining distinctiveness.
    """

    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Temperature for softmax (lower = more peaked)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        score_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encourage high similarity for positive pairs and mutual nearest neighbor property.

        Args:
            z1: (B, K, D) descriptors from image 1
            z2: (B, K, D) descriptors from image 2
            score_weights: Optional (B, K) confidence weights

        Returns:
            Scalar loss
        """
        # Compute similarity matrix (B, K, K)
        similarity = torch.einsum("bkd,bmd->bkm", z1, z2)  # (B, K, K)

        # For each descriptor in image 1, how similar is it to its correspondence in image 2?
        # Diagonal elements are the positive pairs
        # positive_sim = torch.diagonal(similarity, dim1=1, dim2=2)  # (B, K)

        # Compute softmax-based matching score
        # This encourages the positive match to have highest similarity
        max_sim_from_1to2 = torch.softmax(similarity / self.temperature, dim=2)
        max_sim_from_2to1 = torch.softmax(similarity / self.temperature, dim=1)

        # Loss: encourage diagonal to have high probability in both directions
        matching_score_1to2 = max_sim_from_1to2[
            :, range(max_sim_from_1to2.shape[1]), range(max_sim_from_1to2.shape[1])
        ]  # (B, K)
        matching_score_2to1 = max_sim_from_2to1[
            :, range(max_sim_from_2to1.shape[1]), range(max_sim_from_2to1.shape[1])
        ]  # (B, K)

        # Cross-entropy style loss (maximize probability of correct match)
        if score_weights is not None:
            loss = -(
                score_weights
                * (
                    torch.log(matching_score_1to2 + 1e-8)
                    + torch.log(matching_score_2to1 + 1e-8)
                )
            ).sum() / (score_weights.sum() + 1e-8)
        else:
            loss = -(
                torch.log(matching_score_1to2 + 1e-8)
                + torch.log(matching_score_2to1 + 1e-8)
            ).mean()

        return loss


class EpipolarLoss(nn.Module):
    """
    Enforces epipolar geometry constraints on matched keypoints.
    This ensures matches will pass RANSAC geometric verification.
    """

    def __init__(self, threshold: float = 1e-3):
        """
        Args:
            threshold: Threshold for epipolar distance
        """
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        coords1: torch.Tensor,
        coords2: torch.Tensor,
        fundamental_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute epipolar constraint: x2^T F x1 = 0

        Args:
            coords1: (B, K, 2) keypoint coordinates in image 1
            coords2: (B, K, 2) keypoint coordinates in image 2
            fundamental_matrix: (B, 3, 3) fundamental matrix from image 1 to 2

        Returns:
            Scalar epipolar loss
        """
        B, K, _ = coords1.shape

        # Convert to homogeneous coordinates
        ones = torch.ones(B, K, 1, device=coords1.device)
        coords1_h = torch.cat([coords1, ones], dim=-1)  # (B, K, 3)
        coords2_h = torch.cat([coords2, ones], dim=-1)  # (B, K, 3)

        # Compute epipolar error: x2^T @ F @ x1
        # F @ x1: (B, 3, 3) @ (B, K, 3) -> (B, K, 3)
        Fx1 = torch.einsum("bij,bkj->bki", fundamental_matrix, coords1_h)

        # x2^T @ (F @ x1): (B, K, 3) * (B, K, 3) -> (B, K)
        epipolar_error = torch.einsum("bki,bki->bk", coords2_h, Fx1)

        # Normalize by ||Fx1||
        Fx1_norm = torch.norm(Fx1[..., :2], dim=-1, keepdim=True) + 1e-8
        epipolar_distance = torch.abs(epipolar_error) / Fx1_norm.squeeze(-1)

        # Robust loss (similar to Huber)
        loss = torch.where(
            epipolar_distance < self.threshold,
            0.5 * epipolar_distance.pow(2),
            self.threshold * (epipolar_distance - 0.5 * self.threshold),
        )

        return loss.mean()


class PeaknessLoss(nn.Module):
    """
    Encourages detected keypoints to have high confidence scores.
    This improves the reliability of extracted features for matching.
    """

    def __init__(self, target_score: float = 0.8):
        """
        Args:
            target_score: Target score for detected keypoints
        """
        super().__init__()
        self.target_score = target_score

    def forward(
        self,
        score_map: torch.Tensor,
        keypoint_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage high scores at detected keypoint locations.

        Args:
            score_map: (B, 1, H, W) raw score logits
            keypoint_mask: (B, 1, H, W) binary mask of detected keypoints

        Returns:
            Scalar peakness loss
        """
        # Apply sigmoid
        scores = torch.sigmoid(score_map)

        # Only consider locations where keypoints should be detected
        masked_scores = scores * keypoint_mask

        # Count number of keypoints
        num_keypoints = keypoint_mask.sum() + 1e-8

        # Loss: encourage scores at keypoint locations to reach target
        target = torch.ones_like(masked_scores) * self.target_score
        loss = (
            F.mse_loss(masked_scores, target * keypoint_mask, reduction="sum")
            / num_keypoints
        )

        return loss


class TotalLoss(nn.Module):
    """Aggregated loss for keypoint detection and descriptor learning."""

    def __init__(
        self,
        lambda_det: float = 1.0,
        lambda_rot: float = 0.5,
        lambda_desc: float = 1.0,
        margin: float = 0.5,
        lambda_repeatability: float = 0.5,
        lambda_match_count: float = 0.3,
        lambda_epipolar: float = 0.2,
        lambda_peakness: float = 0.1,
    ):
        """
        Args:
            lambda_det: Weight for detector loss
            lambda_rot: Weight for rotation equivariance loss
            lambda_desc: Weight for descriptor loss
            margin: Triplet loss margin
            lambda_repeatability: Weight for repeatability loss
            lambda_match_count: Weight for match count loss
            lambda_epipolar: Weight for epipolar loss
            lambda_peakness: Weight for peakness loss
        """
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_rot = lambda_rot
        self.lambda_desc = lambda_desc
        self.lambda_repeatability = lambda_repeatability
        self.lambda_match_count = lambda_match_count
        self.lambda_epipolar = lambda_epipolar
        self.lambda_peakness = lambda_peakness
        self.detector_loss = DetectorLoss()
        self.rotation_loss = RotationEquivarianceLoss()
        self.descriptor_loss = DescriptorLoss(margin=margin)
        self.repeatability_loss = RepeatabilityLoss()
        self.match_count_loss = MatchCountLoss()
        self.epipolar_loss = EpipolarLoss()
        self.peakness_loss = PeaknessLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Compute total loss.

        Args:
            outputs: Dict containing:
                - score_logits: (B, 1, H, W) keypoint scores
                - score_logits_2: (B, 1, H, W) scores for image 2 (for repeatability)
                - orientations1: (B, K) orientations
                - orientations2: (B, K) orientations
                - z1: (B, K, D) descriptors
                - z2: (B, K, D) descriptors
                - negatives: (B, K, N, D) negatives

            targets: Dict containing:
                - score_gt: (B, 1, H, W) ground truth scores
                - rotation_angle: (B,) rotation angle
                - homography: (B, 3, 3) homography matrix
                - fundamental_matrix: (B, 3, 3) fundamental matrix (if available)
                - invariant_coords: (B, K, 2) keypoint coordinates
                - invariant_coords_2: (B, K, 2) corresponding coordinates in image 2

        Returns:
            Dict with individual and total losses
        """
        losses = {}
        device = outputs["z1"].device
        # 1. Detector loss
        if "score_logits" in outputs and "score_gt" in targets:
            L_det = self.detector_loss(outputs["score_logits"], targets["score_gt"])
            losses["detector"] = L_det
        else:
            L_det = torch.tensor(0.0, device=device)
            losses["detector"] = L_det

        # 2. Rotation loss
        if "rotation_angle" in targets:
            L_rot = self.rotation_loss(
                outputs["orientations1"],
                outputs["orientations2"],
                targets["rotation_angle"],
            )
            losses["rotation"] = L_rot
        else:
            L_rot = torch.tensor(0.0, device=device)
            losses["rotation"] = L_rot

        # 3. Descriptor loss
        if "score_logits" in outputs and "invariant_coords" in targets:
            score_weights = self._sample_scores_at_coords(
                outputs["score_logits"], targets["invariant_coords"]
            )
            score_weights = torch.sigmoid(score_weights)
        else:
            score_weights = None

        L_desc = self.descriptor_loss(
            outputs["z1"],
            outputs["z2"],
            outputs["negatives"],
            weights=score_weights,
        )
        losses["descriptor"] = L_desc

        # 4. Repeatability loss
        if "score_logits_2" in outputs and "homography" in targets:
            L_repeat = self.repeatability_loss(
                outputs["score_logits"],
                outputs["score_logits_2"],
                targets["homography"],
            )
            losses["repeatability"] = L_repeat
        else:
            L_repeat = torch.tensor(0.0, device=device)
            losses["repeatability"] = L_repeat

        # 5. Match count loss
        L_match = self.match_count_loss(
            outputs["z1"],
            outputs["z2"],
            score_weights=score_weights,
        )
        losses["match_count"] = L_match

        # 6. Epipolar loss (if fundamental matrix available)
        if "fundamental_matrix" in targets and "invariant_coords_2" in targets:
            L_epipolar = self.epipolar_loss(
                targets["invariant_coords"],
                targets["invariant_coords_2"],
                targets["fundamental_matrix"],
            )
            losses["epipolar"] = L_epipolar
        else:
            L_epipolar = torch.tensor(0.0, device=device)
            losses["epipolar"] = L_epipolar

        # 7. Peakness loss
        if "score_gt" in targets:
            L_peak = self.peakness_loss(outputs["score_logits"], targets["score_gt"])
            losses["peakness"] = L_peak
        else:
            L_peak = torch.tensor(0.0, device=device)
            losses["peakness"] = L_peak

        # ===== Total Loss =====
        L_total = (
            self.lambda_det * L_det
            + self.lambda_rot * L_rot
            + self.lambda_desc * L_desc
            + self.lambda_repeatability * L_repeat
            + self.lambda_match_count * L_match
            + self.lambda_epipolar * L_epipolar
            + self.lambda_peakness * L_peak
        )
        losses["total"] = L_total

        return losses

    def _sample_scores_at_coords(
        self,
        score_logits: torch.Tensor,
        coords: torch.Tensor,
    ):
        """Sample score logits at specified coordinates."""
        B, _, H, W = score_logits.shape

        # Normalize coordinates to [-1, 1]
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

        grid = grid.unsqueeze(2)
        sampled = F.grid_sample(
            score_logits, grid, mode="bilinear", align_corners=False
        )
        scores = sampled.squeeze(1).squeeze(-1)
        return scores
