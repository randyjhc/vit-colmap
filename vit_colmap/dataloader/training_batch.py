"""
Batch processing for training with positive/negative sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .training_sampler import TrainingSampler


class TrainingBatchProcessor:
    """Process batches for training with positive/negative sampling."""

    def __init__(
        self,
        model: nn.Module,
        sampler: Optional[TrainingSampler] = None,
        descriptor_head: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: ViTFeatureModel instance
            sampler: TrainingSampler instance (created if None)
            descriptor_head: Optional separate descriptor head (uses model's head if None)
        """
        self.model = model
        self.sampler = sampler if sampler is not None else TrainingSampler()
        self.descriptor_head = descriptor_head
        self._projection_matrix: Optional[torch.Tensor] = None

    def extract_rotation_from_homography(self, H: torch.Tensor) -> torch.Tensor:
        """
        Extract rotation angle from homography matrix.

        For a homography H = [[a, b, tx], [c, d, ty], [e, f, 1]],
        the rotation component is approximately atan2(c, a) for small perspective.

        Args:
            H: (B, 3, 3) homography matrices

        Returns:
            rotation_angle: (B,) rotation angles in radians
        """
        # Extract rotation from the top-left 2x2 submatrix
        # This is approximate for perspective transforms
        rotation_angle = torch.atan2(H[:, 1, 0], H[:, 0, 0])
        return rotation_angle

    def project_to_128d(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 768D (or other dim) features to 128D using descriptor head.

        Args:
            features: (B, K, C) features (C=768 for DINOv2)

        Returns:
            descriptors: (B, K, 128) L2-normalized descriptors
        """
        B, K, C = features.shape

        # If we have a separate descriptor head, use it
        if self.descriptor_head is not None:
            # Reshape for conv: (B*K, C, 1, 1)
            features_reshaped = features.view(B * K, C, 1, 1)
            descriptors = self.descriptor_head(features_reshaped)
            descriptors = descriptors.view(B, K, -1)
        else:
            # Use model's descriptor head
            # The model's head expects (B, C, H, W), so reshape accordingly
            features_reshaped = features.permute(0, 2, 1).unsqueeze(-1)  # (B, C, K, 1)

            # Pass through model's trunk and descriptor head
            # Since we already have backbone features, we need to project them
            # Use a simple linear projection as fallback
            if hasattr(self.model, "trunk") and hasattr(self.model, "descriptor_head"):
                # This is a simplification - ideally we'd use the full head
                # For now, use a simple learned projection
                descriptors = self._simple_projection(features, target_dim=128)
            else:
                descriptors = self._simple_projection(features, target_dim=128)

        # L2 normalize
        descriptors = F.normalize(descriptors, p=2, dim=-1)
        return descriptors

    def _simple_projection(
        self,
        features: torch.Tensor,
        target_dim: int = 128,
    ) -> torch.Tensor:
        """
        Simple linear projection to target dimension.

        Args:
            features: (B, K, C) input features
            target_dim: Output dimension

        Returns:
            projected: (B, K, target_dim)
        """
        B, K, C = features.shape
        device = features.device

        # Create projection matrix if not exists
        if self._projection_matrix is None or self._projection_matrix.shape[0] != C:
            # Random orthogonal projection (Xavier initialization)
            self._projection_matrix = torch.randn(
                C, target_dim, device=device, dtype=features.dtype
            )
            self._projection_matrix /= np.sqrt(C)

        # Move to correct device if needed
        if self._projection_matrix.device != device:
            self._projection_matrix = self._projection_matrix.to(device)

        # Project: (B, K, C) @ (C, target_dim) -> (B, K, target_dim)
        projected = features @ self._projection_matrix
        return projected

    def create_score_ground_truth(
        self,
        invariant_coords: torch.Tensor,
        feature_size: Tuple[int, int],
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Create ground truth score heatmap from invariant points.

        Args:
            invariant_coords: (B, K, 2) coordinates in feature map space
            feature_size: (H_p, W_p) feature map size
            sigma: Gaussian sigma for heatmap

        Returns:
            score_gt: (B, 1, H_p, W_p) ground truth heatmap
        """
        B, K, _ = invariant_coords.shape
        H_p, W_p = feature_size
        device = invariant_coords.device

        # Create coordinate grids
        y_coords = torch.arange(H_p, device=device, dtype=torch.float32)
        x_coords = torch.arange(W_p, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")  # (H_p, W_p)

        # Compute Gaussian heatmap
        score_gt = torch.zeros(B, 1, H_p, W_p, device=device)

        for b in range(B):
            for k in range(K):
                cx, cy = invariant_coords[b, k]
                # Gaussian centered at (cx, cy)
                gaussian = torch.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
                )
                score_gt[b, 0] = torch.maximum(score_gt[b, 0], gaussian)

        return score_gt

    def sample_orientations_at_coords(
        self,
        keypoint_outputs: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample orientations from keypoint head output at specified coordinates.

        Args:
            keypoint_outputs: (B, 4, H, W) - [score, dx, dy, orientation]
            coords: (B, K, 2) coordinates

        Returns:
            orientations: (B, K) sampled orientations
        """
        B, _, H, W = keypoint_outputs.shape

        # Extract orientation channel
        orientations_map = keypoint_outputs[:, 3:4, :, :]  # (B, 1, H, W)

        # Normalize coordinates for grid_sample
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
        grid = grid.unsqueeze(2)  # (B, K, 1, 2)

        # Sample
        sampled = F.grid_sample(
            orientations_map, grid, mode="bilinear", align_corners=False
        )
        orientations = sampled.squeeze(1).squeeze(-1)  # (B, K)

        return orientations

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a batch of image pairs for training.

        Args:
            batch: Dict from HPatchesDataset containing:
                - img1: (B, 3, H, W)
                - img2: (B, 3, H, W)
                - H: (B, 3, 3) homography

        Returns:
            outputs: Dict containing model predictions
            targets: Dict containing ground truth and generated samples
        """
        img1 = batch["img1"]
        img2 = batch["img2"]
        H = batch["H"]

        B, _, img_H, img_W = img1.shape
        image_size = (img_H, img_W)

        # 1. Extract backbone features ONCE (frozen, use inference_mode for better perf)
        with torch.inference_mode():
            features1 = self.model._extract_backbone_features(
                img1
            )  # (B, 768, H_p, W_p)
            features2 = self.model._extract_backbone_features(img2)

        _, C, H_p, W_p = features1.shape
        feature_size = (H_p, W_p)

        # 2. Select invariant points based on cosine similarity
        invariant_coords = self.sampler.select_invariant_points(
            features1, features2, H, image_size
        )  # (B, K, 2) in features2 space

        # 3. Generate positive pairs
        f1, f2 = self.sampler.generate_positive_pairs(
            features1, features2, H, invariant_coords, image_size
        )  # (B, K, 768)

        # 4. Project to 128D descriptors
        z1 = self.project_to_128d(f1)  # (B, K, 128)
        z2 = self.project_to_128d(f2)  # (B, K, 128)

        # 5. Generate negative examples
        negatives = self.sampler.generate_all_negatives(
            features1, features2, f2, invariant_coords
        )  # (B, K, N, 768)

        # Project negatives to 128D
        B_n, K_n, N_n, C_n = negatives.shape
        negatives_flat = negatives.view(B_n, K_n * N_n, C_n)
        negatives_128d_flat = self.project_to_128d(negatives_flat)
        negatives_128d = negatives_128d_flat.view(B_n, K_n, N_n, 128)

        # 6. Forward pass through trainable heads only (REUSE backbone features)
        # This avoids redundant backbone passes - major speedup!
        outputs1_full = self.model.forward_from_backbone_features(features1)
        outputs2_full = self.model.forward_from_backbone_features(features2)

        # 7. Sample orientations at invariant point locations
        # Map invariant coords (in features2 space) to output space (1/4 resolution)
        # Feature space is 1/14, output space is 1/4, so scale by 14/4 = 3.5
        scale_factor = self.sampler.patch_size / 4.0
        output_coords = invariant_coords * scale_factor  # Scale to 1/4 resolution space

        orientations2 = self.sample_orientations_at_coords(
            outputs2_full["keypoints"], output_coords
        )

        # Map to image1's output space
        H_inv = torch.linalg.inv(H)
        coords_in_img1_feature = self.sampler.transform_coords_with_homography(
            invariant_coords,
            H_inv,
            from_feature_space=True,
            image_size=image_size,
            feature_size=feature_size,
        )
        output_coords_img1 = coords_in_img1_feature * scale_factor

        orientations1 = self.sample_orientations_at_coords(
            outputs1_full["keypoints"], output_coords_img1
        )

        # 8. Extract rotation angle from homography
        rotation_angle = self.extract_rotation_from_homography(H)

        # 9. Create ground truth score heatmap
        # Scale invariant coords to output space (1/2 resolution)
        score_gt = self.create_score_ground_truth(
            output_coords,
            (outputs2_full["keypoints"].shape[2], outputs2_full["keypoints"].shape[3]),
        )

        # Assemble outputs and targets
        outputs = {
            "z1": z1,  # (B, K, 128)
            "z2": z2,  # (B, K, 128)
            "negatives": negatives_128d,  # (B, K, N, 128)
            "orientations1": orientations1,  # (B, K)
            "orientations2": orientations2,  # (B, K)
            "score_logits": outputs2_full["keypoints"][:, 0:1, :, :],  # (B, 1, H, W)
            "keypoints1_full": outputs1_full["keypoints"],
            "keypoints2_full": outputs2_full["keypoints"],
            "descriptors1_full": outputs1_full["descriptors"],
            "descriptors2_full": outputs2_full["descriptors"],
        }

        targets = {
            "invariant_coords": output_coords,  # (B, K, 2) in output space
            "rotation_angle": rotation_angle,  # (B,)
            "score_gt": score_gt,  # (B, 1, H, W)
            "H": H,
            "image_size": image_size,
            "feature_size": feature_size,
        }

        return outputs, targets


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of dicts from HPatchesDataset

    Returns:
        Collated dict with batched tensors
    """
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        elif isinstance(batch[0][key], str):
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = [item[key] for item in batch]
    return collated


class EnhancedTrainingBatchProcessor:
    """
    Enhanced batch processor that provides data for all loss components.

    Modifications from original:
    1. Runs model on both images (not just image1)
    2. Computes fundamental matrix from homography
    3. Provides warped coordinates for image2
    """

    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler

    def process_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process batch to prepare data for training.

        Args:
            batch: Dictionary containing:
                - image1: (B, 3, H, W) first image
                - image2: (B, 3, H, W) second image (warped)
                - homography: (B, 3, 3) homography from image1 to image2
                - rotation_angle: (B,) rotation angle

        Returns:
            outputs: Dictionary for model outputs
            targets: Dictionary for loss targets
        """
        # ===== MODIFIED: Run model on BOTH images =====
        with torch.inference_mode():
            outputs1 = self.model(batch["image1"])
            outputs2 = self.model(batch["image2"])

        # Extract feature maps
        keypoints_map1 = outputs1["keypoints"]  # (B, 4, H, W)
        descriptors_map1 = outputs1["descriptors"]  # (B, D, H, W)

        keypoints_map2 = outputs2["keypoints"]
        descriptors_map2 = outputs2["descriptors"]

        B, _, H, W = keypoints_map1.shape
        invariant_coords, score_gt = self.sampler.sample_invariant_points(
            batch["homography"], (H, W)
        )

        # Warp coordinates to image2
        invariant_coords_warped = self._warp_coordinates(
            invariant_coords, batch["homography"]
        )

        # Sample features at invariant points
        sampled_z1 = self._sample_descriptors_at_coords(
            descriptors_map1, invariant_coords
        )
        sampled_z2 = self._sample_descriptors_at_coords(
            descriptors_map2, invariant_coords_warped
        )

        # Sample orientations
        sampled_orientations1 = self._sample_at_coords(
            keypoints_map1[:, 3:4], invariant_coords
        ).squeeze(1)
        sampled_orientations2 = self._sample_at_coords(
            keypoints_map2[:, 3:4], invariant_coords_warped
        ).squeeze(1)

        # Sample negative descriptors (same as before)
        negatives = self.sampler.sample_negatives(
            descriptors_map1,
            invariant_coords,
            keypoints_map1[:, 0:1],
        )

        # ===== Compute fundamental matrix =====
        fundamental_matrix = self._homography_to_fundamental(
            batch["homography"],
            image_size=(H * 4, W * 4),  # Original image size (before downsampling)
        )

        # ===== Prepare outputs =====
        outputs = {
            "score_logits": keypoints_map1[:, 0:1],  # (B, 1, H, W)
            "score_logits_2": keypoints_map2[:, 0:1],  # (B, 1, H, W)
            "orientations1": sampled_orientations1,  # (B, K)
            "orientations2": sampled_orientations2,  # (B, K)
            "z1": sampled_z1,  # (B, K, D)
            "z2": sampled_z2,  # (B, K, D)
            "negatives": negatives,  # (B, K, N, D)
        }

        # ===== Prepare targets =====
        targets = {
            "score_gt": score_gt,  # (B, 1, H, W)
            "rotation_angle": batch["rotation_angle"],  # (B,)
            "homography": batch["homography"],  # (B, 3, 3)
            "fundamental_matrix": fundamental_matrix,  # (B, 3, 3)
            "invariant_coords": invariant_coords,  # (B, K, 2)
            "invariant_coords_2": invariant_coords_warped,  # (B, K, 2)
        }

        return outputs, targets

    def _warp_coordinates(
        self, coords: torch.Tensor, homography: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp coordinates using homography.

        Args:
            coords: (B, K, 2) coordinates (x, y)
            homography: (B, 3, 3) homography matrix

        Returns:
            warped_coords: (B, K, 2) warped coordinates
        """
        B, K, _ = coords.shape

        # Convert to homogeneous coordinates
        ones = torch.ones(B, K, 1, device=coords.device)
        coords_h = torch.cat([coords, ones], dim=-1)  # (B, K, 3)

        # Apply homography: H @ [x, y, 1]^T
        coords_warped_h = torch.einsum("bij,bkj->bki", homography, coords_h)

        # Normalize by z coordinate
        coords_warped = coords_warped_h[..., :2] / (coords_warped_h[..., 2:3] + 1e-8)

        return coords_warped

    def _homography_to_fundamental(
        self,
        H: torch.Tensor,
        image_size: Tuple[int, int],
        intrinsic_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Approximate fundamental matrix from homography.

        For planar scenes: F = K^-T [t]_x H K^-1
        We use a simplified approximation suitable for training.

        Args:
            H: (B, 3, 3) homography matrix
            image_size: (height, width) of images
            intrinsic_scale: Scale factor for intrinsics

        Returns:
            F: (B, 3, 3) fundamental matrix
        """
        B = H.shape[0]
        device = H.device
        h, w = image_size

        # Simple camera intrinsics (assume focal length = max(w, h))
        f = max(h, w) * intrinsic_scale
        cx, cy = w / 2, h / 2

        K = (
            torch.tensor(
                [[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=H.dtype, device=device
            )
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        K_inv = torch.inverse(K)

        # Decompose homography: H = R + t*n^T
        # For small motions, we approximate the epipole direction
        # Simplified: Use the translation part of H

        # Extract translation direction (rightmost column of H after normalization)
        t = H[:, :, 2].unsqueeze(-1)  # (B, 3, 1)

        # Create skew-symmetric matrix [t]_x
        tx = self._skew_symmetric(t.squeeze(-1))  # (B, 3, 3)

        # Fundamental matrix: F = K^-T [t]_x H K^-1
        F = K_inv.transpose(-2, -1) @ tx @ H @ K_inv

        # Normalize: Make F have unit Frobenius norm
        F_norm = torch.norm(F, dim=(-2, -1), keepdim=True) + 1e-8
        F = F / F_norm

        # Enforce rank-2 constraint (optional, can be skipped for efficiency)
        # F = self._enforce_rank2(F)

        return F

    def _skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """
        Create skew-symmetric matrix from vector.

        [v]_x = [[0, -v3, v2],
                 [v3, 0, -v1],
                 [-v2, v1, 0]]

        Args:
            v: (B, 3) vector

        Returns:
            (B, 3, 3) skew-symmetric matrix
        """
        B = v.shape[0]
        device = v.device

        zeros = torch.zeros(B, device=device)

        skew = torch.stack(
            [
                torch.stack([zeros, -v[:, 2], v[:, 1]], dim=-1),
                torch.stack([v[:, 2], zeros, -v[:, 0]], dim=-1),
                torch.stack([-v[:, 1], v[:, 0], zeros], dim=-1),
            ],
            dim=1,
        )

        return skew

    def _enforce_rank2(self, F: torch.Tensor) -> torch.Tensor:
        """
        Enforce rank-2 constraint on fundamental matrix using SVD.

        Args:
            F: (B, 3, 3) fundamental matrix

        Returns:
            F_rank2: (B, 3, 3) rank-2 fundamental matrix
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(F)

        # Set smallest singular value to zero
        S[:, 2] = 0

        # Reconstruct
        F_rank2 = U @ torch.diag_embed(S) @ Vh

        return F_rank2

    def _sample_descriptors_at_coords(
        self,
        descriptor_map: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Sample descriptors at specified coordinates."""
        B, D, H, W = descriptor_map.shape

        # Normalize coordinates to [-1, 1]
        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

        # Add batch dimension for grid_sample
        grid = grid.unsqueeze(2)  # (B, K, 1, 2)

        # Sample
        sampled = F.grid_sample(
            descriptor_map,
            grid,
            mode="bilinear",
            align_corners=False,
        )

        # Reshape to (B, K, D)
        sampled = sampled.squeeze(-1).transpose(1, 2)

        return sampled

    def _sample_at_coords(
        self,
        feature_map: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Sample arbitrary feature map at coordinates."""
        B, C, H, W = feature_map.shape

        grid = coords.clone()
        grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
        grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

        grid = grid.unsqueeze(2)

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            align_corners=False,
        )

        sampled = sampled.squeeze(-1).transpose(1, 2)

        return sampled
